//! Backend codegen for remaining integer types: i8, u8, i16, u16, u64.
//!
//! Uses a unified IntVecType to generate all 10 types (2 widths each).

use indoc::formatdoc;

// ============================================================================
// Data Model
// ============================================================================

/// A generic integer vector type for backend generation.
#[derive(Clone, Debug)]
pub(super) struct IntVecType {
    /// Element type: "i8", "u8", "i16", "u16", "u64"
    pub elem: &'static str,
    /// Whether the type is signed
    pub signed: bool,
    /// Element size in bits
    pub elem_bits: usize,
    /// Number of lanes
    pub lanes: usize,
    /// Width in bits (128, 256)
    pub width_bits: usize,
}

impl IntVecType {
    /// Type name: "i8x16", "u16x8", etc.
    pub fn name(&self) -> String {
        format!("{}x{}", self.elem, self.lanes)
    }

    /// Trait name: "I8x16Backend", "U16x8Backend", etc.
    pub fn trait_name(&self) -> String {
        let upper = match self.elem {
            "i8" => "I8",
            "u8" => "U8",
            "i16" => "I16",
            "u16" => "U16",
            "u64" => "U64",
            _ => unreachable!(),
        };
        format!("{upper}x{}Backend", self.lanes)
    }

    /// Array type: "[i8; 16]", "[u16; 8]", etc.
    fn array_type(&self) -> String {
        format!("[{}; {}]", self.elem, self.lanes)
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

    /// x86 suffix for set1: "epi8", "epi16", "epi64x"
    fn x86_set1_suffix(&self) -> &'static str {
        match self.elem_bits {
            8 => "epi8",
            16 => "epi16",
            64 => "epi64x",
            _ => unreachable!(),
        }
    }

    /// x86 suffix for arithmetic ops: "epi8", "epi16", "epi64"
    fn x86_arith_suffix(&self) -> &'static str {
        match self.elem_bits {
            8 => "epi8",
            16 => "epi16",
            64 => "epi64",
            _ => unreachable!(),
        }
    }

    /// x86 suffix for min/max: signed uses epi*, unsigned uses epu*
    fn x86_minmax_suffix(&self) -> &'static str {
        match (self.signed, self.elem_bits) {
            (true, 8) => "epi8",
            (false, 8) => "epu8",
            (true, 16) => "epi16",
            (false, 16) => "epu16",
            (_, 64) => unreachable!("u64 min/max is polyfilled"),
            _ => unreachable!(),
        }
    }

    /// Whether this type has native min/max (not polyfilled)
    fn has_native_minmax(&self) -> bool {
        self.elem_bits != 64
    }

    /// Whether this type has native multiply
    fn has_native_mul(&self) -> bool {
        self.elem_bits == 16 // Only i16/u16 have mullo_epi16
    }

    /// Whether the uniform-variable-shift and saturating-arithmetic families
    /// are emitted for this element width.
    ///
    /// x86 has no `_mm*_adds_epi32`/`_subs_epi32` at any tier and wasm128 has
    /// no 32/64-bit `*_add_sat`; only NEON does. Exposing the op above 16 bits
    /// would ship a "primitive" that is one instruction on NEON and a 2-3 op
    /// emulation everywhere else. See `docs/CROSS-ISA-INT-PRIMITIVES.md`.
    fn has_saturating(&self) -> bool {
        self.elem_bits == 8 || self.elem_bits == 16
    }

    /// Whether this type is native on NEON (128-bit only)
    fn native_on_neon(&self) -> bool {
        self.width_bits == 128
    }

    /// Whether this type is native on WASM (128-bit only)
    fn native_on_wasm(&self) -> bool {
        self.width_bits == 128
    }

    /// NEON native 128-bit type
    fn neon_native_type(&self) -> String {
        match (self.signed, self.elem_bits) {
            (true, 8) => "int8x16_t".to_string(),
            (false, 8) => "uint8x16_t".to_string(),
            (true, 16) => "int16x8_t".to_string(),
            (false, 16) => "uint16x8_t".to_string(),
            (false, 64) => "uint64x2_t".to_string(),
            _ => unreachable!(),
        }
    }

    /// NEON intrinsic suffix: "s8", "u8", "s16", "u16", "u64"
    fn neon_suffix(&self) -> &'static str {
        match (self.signed, self.elem_bits) {
            (true, 8) => "s8",
            (false, 8) => "u8",
            (true, 16) => "s16",
            (false, 16) => "u16",
            (false, 64) => "u64",
            _ => unreachable!(),
        }
    }

    /// NEON unsigned type for comparison results
    fn neon_unsigned_type(&self) -> &'static str {
        match self.elem_bits {
            8 => "uint8x16_t",
            16 => "uint16x8_t",
            64 => "uint64x2_t",
            _ => unreachable!(),
        }
    }

    /// WASM intrinsic prefix for this element type: "i8x16", "u8x16", "i16x8", etc.
    fn wasm_prefix(&self) -> String {
        format!("{}x{}", self.elem, self.lanes_per_128())
    }

    /// Lanes in a 128-bit sub-vector
    fn lanes_per_128(&self) -> usize {
        128 / self.elem_bits
    }

    /// The signed counterpart of this element type (for bias trick)
    fn signed_elem(&self) -> &'static str {
        match self.elem_bits {
            8 => "i8",
            16 => "i16",
            64 => "i64",
            _ => unreachable!(),
        }
    }

    /// The cast expression to convert splat value: "" for signed, "as i8"/"as i16"/"as i64" for unsigned
    fn x86_set1_cast(&self) -> &'static str {
        if self.signed {
            ""
        } else {
            match self.elem_bits {
                8 => " as i8",
                16 => " as i16",
                64 => " as i64",
                _ => unreachable!(),
            }
        }
    }

    /// Signed type suffix for literal constants (e.g., `-1_i8` instead of `-1 as i8`)
    fn x86_set1_literal_suffix(&self) -> &'static str {
        if self.signed {
            ""
        } else {
            match self.elem_bits {
                8 => "_i8",
                16 => "_i16",
                64 => "_i64",
                _ => unreachable!(),
            }
        }
    }

    /// Bias value for unsigned comparison (as signed literal)
    fn unsigned_bias_literal(&self) -> &'static str {
        match self.elem_bits {
            8 => "i8::MIN",
            16 => "i16::MIN",
            64 => "i64::MIN",
            _ => unreachable!(),
        }
    }
}

/// All remaining integer types to generate backends for.
pub(super) fn all_remaining_int_types() -> Vec<IntVecType> {
    vec![
        // i8
        IntVecType {
            elem: "i8",
            signed: true,
            elem_bits: 8,
            lanes: 16,
            width_bits: 128,
        },
        IntVecType {
            elem: "i8",
            signed: true,
            elem_bits: 8,
            lanes: 32,
            width_bits: 256,
        },
        // u8
        IntVecType {
            elem: "u8",
            signed: false,
            elem_bits: 8,
            lanes: 16,
            width_bits: 128,
        },
        IntVecType {
            elem: "u8",
            signed: false,
            elem_bits: 8,
            lanes: 32,
            width_bits: 256,
        },
        // i16
        IntVecType {
            elem: "i16",
            signed: true,
            elem_bits: 16,
            lanes: 8,
            width_bits: 128,
        },
        IntVecType {
            elem: "i16",
            signed: true,
            elem_bits: 16,
            lanes: 16,
            width_bits: 256,
        },
        // u16
        IntVecType {
            elem: "u16",
            signed: false,
            elem_bits: 16,
            lanes: 8,
            width_bits: 128,
        },
        IntVecType {
            elem: "u16",
            signed: false,
            elem_bits: 16,
            lanes: 16,
            width_bits: 256,
        },
        // u64
        IntVecType {
            elem: "u64",
            signed: false,
            elem_bits: 64,
            lanes: 2,
            width_bits: 128,
        },
        IntVecType {
            elem: "u64",
            signed: false,
            elem_bits: 64,
            lanes: 4,
            width_bits: 256,
        },
    ]
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

pub(super) fn generate_x86_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_int_impl(ty));
        code.push('\n');
    }
    code
}

fn generate_x86_int_impl(ty: &IntVecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "X64V3Token");
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let elem = ty.elem;
    let set1_suf = ty.x86_set1_suffix();
    let arith_suf = ty.x86_arith_suffix();
    let set1_cast = ty.x86_set1_cast();
    let set1_lit = ty.x86_set1_literal_suffix();

    let mut body = String::new();

    // Construction
    body.push_str(&formatdoc! {r#"
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                {p}_set1_{set1_suf}(v{set1_cast})
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
                {p}_add_{arith_suf}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_{arith_suf}(a, b)
            }}
            {baseline_end}
    "#});

    // Mul
    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mullo_epi16(a, b)
            }}
        "#});
    }

    // Neg
    if ty.signed {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_sub_{arith_suf}({p}_setzero_si{bits}(), a)
            }}
        "#});
    }

    // Math: min/max
    if ty.has_native_minmax() {
        let mm_suf = ty.x86_minmax_suffix();
        body.push_str(&formatdoc! {r#"

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_min_{mm_suf}(a, b)
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_max_{mm_suf}(a, b)
            }}
        "#});
    } else {
        // u64: polyfill via bias trick + signed compare + blend
        body.push_str(&formatdoc! {r#"

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                let bias = {p}_set1_epi64x(i64::MIN);
                let a_biased = {p}_xor_si{bits}(a, bias);
                let b_biased = {p}_xor_si{bits}(b, bias);
                let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                {p}_blendv_epi8(a, b, mask)
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                let bias = {p}_set1_epi64x(i64::MIN);
                let a_biased = {p}_xor_si{bits}(a, bias);
                let b_biased = {p}_xor_si{bits}(b, bias);
                let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                {p}_blendv_epi8(b, a, mask)
            }}
        "#});
    }

    // Abs
    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                {p}_abs_{arith_suf}(a)
            }}
        "#});
    }

    // Comparisons
    let cmp_suf = ty.x86_arith_suffix(); // eq/gt intrinsics use same suffix
    if ty.signed {
        // Signed: direct cmpgt
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_{cmp_suf}(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_{cmp_suf}(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_{set1_suf}(-1))
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_{cmp_suf}(b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = {p}_cmpgt_{cmp_suf}(a, b);
                {p}_andnot_si{bits}(gt, {p}_set1_{set1_suf}(-1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_{cmp_suf}(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = {p}_cmpgt_{cmp_suf}(b, a);
                {p}_andnot_si{bits}(lt, {p}_set1_{set1_suf}(-1))
            }}
        "#});
    } else {
        // Unsigned: bias trick for ordering comparisons
        let bias_lit = ty.unsigned_bias_literal();
        let bias_set1 = match ty.elem_bits {
            8 => format!("{p}_set1_epi8({bias_lit})"),
            16 => format!("{p}_set1_epi16({bias_lit})"),
            64 => format!("{p}_set1_epi64x({bias_lit})"),
            _ => unreachable!(),
        };
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_{cmp_suf}(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_{cmp_suf}(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_{set1_suf}(-1{set1_lit}))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                let bias = {bias_set1};
                let sa = {p}_xor_si{bits}(a, bias);
                let sb = {p}_xor_si{bits}(b, bias);
                {p}_cmpgt_{cmp_suf}(sa, sb)
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(_self, b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = <Self as {trait_name}>::simd_gt(_self, a, b);
                {p}_andnot_si{bits}(gt, {p}_set1_{set1_suf}(-1{set1_lit}))
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = <Self as {trait_name}>::simd_gt(_self, b, a);
                {p}_andnot_si{bits}(lt, {p}_set1_{set1_suf}(-1{set1_lit}))
            }}
        "#});
    }

    // Blend
    body.push_str(&formatdoc! {r#"

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_epi8(if_false, if_true, mask)
            }}
    "#});

    // Reduce add
    let reduce_body = generate_x86_int_reduce_add(ty);
    body.push_str(&formatdoc! {r#"

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
        {reduce_body}
            }}
    "#});

    // Bitwise
    body.push_str(&formatdoc! {r#"

            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                {p}_andnot_si{bits}(a, {p}_set1_{set1_suf}(-1{set1_lit}))
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
    "#});

    // Shifts
    body.push_str(&generate_x86_int_shifts(ty));

    // Uniform variable shifts + saturating arithmetic
    body.push_str(&generate_x86_int_uniform_shifts(ty));
    body.push_str(&generate_x86_int_saturating(ty));

    // Boolean
    body.push_str(&generate_x86_int_boolean(ty));

    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
        {body}
        {integer_methods}
        }}
    "#}
}

fn generate_x86_int_reduce_add(ty: &IntVecType) -> String {
    let elem = ty.elem;
    let trait_name = ty.trait_name();
    // For all these types, use scalar fallback (wrapping_add fold).
    // The SIMD horizontal reductions for 8/16-bit are complex and rarely
    // performance-critical. The to_array + fold approach is simple and correct.
    formatdoc! {"
                let arr = <Self as {trait_name}>::to_array(_self, a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)"}
}

fn generate_x86_int_shifts(ty: &IntVecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;

    if ty.elem_bits == 8 {
        // 8-bit: polyfill via 16-bit shift + mask
        let mut code = formatdoc! {r#"

            // ====== Shifts (polyfill via 16-bit) ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                let shifted = {p}_slli_epi16::<N>(a);
                let mask = {p}_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
                {p}_and_si{bits}(shifted, mask)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                let shifted = {p}_srli_epi16::<N>(a);
                let mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                {p}_and_si{bits}(shifted, mask)
            }}
        "#};

        if ty.signed {
            code.push_str(&formatdoc! {r#"

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                let shifted = {p}_srli_epi16::<N>(a);
                let byte_mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                let logical = {p}_and_si{bits}(shifted, byte_mask);
                let zero = {p}_setzero_si{bits}();
                let sign = {p}_cmpgt_epi8(zero, a);
                // High-N-bits fill mask via u16 shift: 0x00 at N == 0 (identity
                // shift needs no sign fill). A u8 `0xFF << (8 - N)` would wrap
                // the shift amount at N == 0 and corrupt negative lanes.
                let fill = {p}_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
                {p}_or_si{bits}(logical, {p}_and_si{bits}(sign, fill))
            }}
            "#});
        }
        code
    } else if ty.elem_bits == 16 {
        // 16-bit: native shifts
        let mut code = formatdoc! {r#"

            // ====== Shifts ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_slli_epi16::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srli_epi16::<N>(a)
            }}
        "#};

        if ty.signed {
            code.push_str(&formatdoc! {r#"

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srai_epi16::<N>(a)
            }}
            "#});
        }
        code
    } else {
        // 64-bit: native shifts (same as i64)
        formatdoc! {r#"

            // ====== Shifts ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_slli_epi64::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srli_epi64::<N>(a)
            }}
        "#}
    }
}

/// Uniform (one runtime count for every lane) shifts, x86.
///
/// `_mm_cvtsi32_si128` writes the count into lane 0 of an XMM and zeroes the
/// rest, so the low quadword the PSLLW/PSRLW/PSRAW family reads as its
/// unsigned 64-bit count is the zero-extended `u32`. That means the hardware
/// already implements the portable contract for out-of-range counts:
/// PSLLW/PSRLW give 0 when the count exceeds the lane width, and PSRAW clamps
/// to a sign fill. No extra clamp is emitted on x86.
fn generate_x86_int_uniform_shifts(ty: &IntVecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    if ty.elem_bits != 16 {
        return String::new();
    }
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();

    let suf = "epi16";
    let mut code = formatdoc! {r#"

            // ====== Uniform variable shifts ======

            {arcane}
            fn shl_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sll_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_srl_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}
        "#};

    if ty.signed {
        code.push_str(&formatdoc! {r#"

            {arcane}
            fn shr_arithmetic_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sra_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}
            "#});
    }
    code
}

/// Saturating add/sub, x86. Emitted only at 8- and 16-bit, where
/// PADDS/PADDUS/PSUBS/PSUBUS exist at every tier (SSE2 / AVX2 / AVX-512BW).
fn generate_x86_int_saturating(ty: &IntVecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    if !ty.has_saturating() {
        return String::new();
    }
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let suf = ty.x86_minmax_suffix(); // epi8/epu8/epi16/epu16 — same signed/unsigned split

    formatdoc! {r#"

            // ====== Saturating arithmetic ======

            {arcane}
            fn saturating_add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_adds_{suf}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_subs_{suf}(a, b)
            }}
    "#}
}

fn generate_x86_int_boolean(ty: &IntVecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;

    match ty.elem_bits {
        8 => {
            // 8-bit: movemask_epi8 gives one bit per byte lane
            let all_mask = if ty.width_bits == 128 {
                "0xFFFF_u32 as i32".to_string()
            } else {
                "-1_i32".to_string() // 0xFFFFFFFF for 32 lanes
            };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_epi8(a) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_epi8(a) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_epi8(a) as u32
            }}
            "#}
        }
        16 => {
            // 16-bit: the byte movemask has one bit per byte. A lane's sign bit is
            // bit 7 of its high byte, which lands on the odd bits (little-endian),
            // so all_true/any_true look only at those. The low byte's bit 7 is
            // not a sign bit. bitmask extracts one bit per 16-bit lane.
            let hi_bytes = if ty.width_bits == 128 {
                "0xAAAA_u32"
            } else {
                "0xAAAA_AAAA_u32"
            };
            let bitmask_body = if ty.width_bits == 128 {
                formatdoc! {r#"
                    let shifted = _mm_srai_epi16::<15>(a);
                    let packed = _mm_packs_epi16(shifted, shifted);
                    (_mm_movemask_epi8(packed) & 0xFF) as u32
                "#}
            } else {
                // 256-bit: _mm256_packs_epi16 interleaves within 128-bit lanes,
                // so extract halves first, then use 128-bit pack for correct order.
                formatdoc! {r#"
                    let shifted = _mm256_srai_epi16::<15>(a);
                    let lo = _mm256_castsi256_si128(shifted);
                    let hi = _mm256_extracti128_si256::<1>(shifted);
                    let packed = _mm_packs_epi16(lo, hi);
                    (_mm_movemask_epi8(packed) as u32) & 0xFFFF
                "#}
            };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                (({p}_movemask_epi8(a) as u32) & {hi_bytes}) == {hi_bytes}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                (({p}_movemask_epi8(a) as u32) & {hi_bytes}) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {bitmask_body}
            }}
            "#}
        }
        64 => {
            // 64-bit: use movemask_pd
            let all_mask = if ty.width_bits == 128 { "0x3" } else { "0xF" };
            formatdoc! {r#"

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
            "#}
        }
        _ => unreachable!(),
    }
}

// ============================================================================
// NEON Implementation Generation
// ============================================================================

pub(super) fn generate_neon_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.native_on_neon() {
            code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
            code.push_str(&generate_neon_native_int_impl(ty));
            code.push('\n');
        }
    }
    code
}

fn generate_neon_native_int_impl(ty: &IntVecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "NeonToken");
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem;
    let ns = ty.neon_suffix();
    let nt = ty.neon_native_type();
    let _ut = ty.neon_unsigned_type();

    // Reinterpret helpers
    let from_u = if ty.signed {
        format!("vreinterpretq_{ns}_u{}", &ns[1..]) // s8_u8, s16_u16
    } else {
        String::new()
    };
    let to_u = if ty.signed {
        format!("vreinterpretq_u{}_{ns}", &ns[1..]) // u8_s8, u16_s16
    } else {
        String::new()
    };

    let wrap_cmp = |intrinsic: &str| -> String {
        if ty.signed {
            format!("{from_u}({intrinsic}(a, b))")
        } else {
            format!("{intrinsic}(a, b)")
        }
    };

    let wrap_ne = if ty.signed {
        format!("{from_u}(vmvnq_u{}(vceqq_{ns}(a, b)))", &ns[1..])
    } else if ty.elem_bits == 64 {
        // NEON lacks vmvnq_u64 — use XOR with all-ones
        "veorq_u64(vceqq_u64(a, b), vdupq_n_u64(u64::MAX))".to_string()
    } else {
        format!("vmvnq_{ns}(vceqq_{ns}(a, b))")
    };

    let blend_body = if ty.signed {
        format!("vbslq_{ns}({to_u}(mask), if_true, if_false)")
    } else {
        format!("vbslq_{ns}(mask, if_true, if_false)")
    };

    let not_body = format!("vmvnq_{ns}(a)");

    // For u64, there's no vmvnq_u64, need different approach
    let not_body = if ty.elem_bits == 64 {
        format!("veorq_{ns}(a, vdupq_n_{ns}(u64::MAX))")
    } else {
        not_body
    };

    // Right shifts lower through vshlq with a negated count: vshrq_n_* rejects
    // N == 0 (immediate encoding has no shift-0 form), while vshlq_* treats a
    // negative count as a right shift and 0 as the identity. LLVM folds the
    // constant splat to the immediate sshr/ushr forms.
    let shr_logical = if ty.signed {
        // Signed logical shift right: cast to unsigned, shift, cast back
        let us = &ns[1..]; // "8", "16"
        format!(
            "vreinterpretq_{ns}_u{us}(vshlq_u{us}(vreinterpretq_u{us}_{ns}(a), vdupq_n_s{us}((-N) as i{us})))"
        )
    } else {
        let us = &ns[1..]; // "8", "16", "64"
        format!("vshlq_{ns}(a, vdupq_n_s{us}((-N) as i{us}))")
    };

    let mut body = formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {nt};

            {arcane}
            fn splat(self, v: {elem}) -> {nt} {{
                vdupq_n_{ns}(v)
            }}

            {arcane}
            fn zero(self) -> {nt} {{
                vdupq_n_{ns}(0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {nt} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {nt} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {nt}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {nt}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: {nt}, b: {nt}) -> {nt} {{ vaddq_{ns}(a, b) }}
            {arcane}
            fn sub(self, a: {nt}, b: {nt}) -> {nt} {{ vsubq_{ns}(a, b) }}
    "#};

    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn mul(self, a: {nt}, b: {nt}) -> {nt} {{ vmulq_{ns}(a, b) }}
        "#});
    }

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn neg(self, a: {nt}) -> {nt} {{ vnegq_{ns}(a) }}
        "#});
    }

    if ty.elem_bits == 64 {
        // NEON lacks vminq_u64/vmaxq_u64 — polyfill with compare + blend
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn min(self, a: {nt}, b: {nt}) -> {nt} {{ vbslq_u64(vcltq_u64(a, b), a, b) }}
            {arcane}
            fn max(self, a: {nt}, b: {nt}) -> {nt} {{ vbslq_u64(vcgtq_u64(a, b), a, b) }}
        "#});
    } else {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn min(self, a: {nt}, b: {nt}) -> {nt} {{ vminq_{ns}(a, b) }}
            {arcane}
            fn max(self, a: {nt}, b: {nt}) -> {nt} {{ vmaxq_{ns}(a, b) }}
        "#});
    }

    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn abs(self, a: {nt}) -> {nt} {{ vabsq_{ns}(a) }}
        "#});
    }

    // Comparisons
    body.push_str(&formatdoc! {r#"

            {arcane}
            fn simd_eq(self, a: {nt}, b: {nt}) -> {nt} {{
                {eq}
            }}
            {arcane}
            fn simd_ne(self, a: {nt}, b: {nt}) -> {nt} {{
                {ne}
            }}
            {arcane}
            fn simd_lt(self, a: {nt}, b: {nt}) -> {nt} {{
                {lt}
            }}
            {arcane}
            fn simd_le(self, a: {nt}, b: {nt}) -> {nt} {{
                {le}
            }}
            {arcane}
            fn simd_gt(self, a: {nt}, b: {nt}) -> {nt} {{
                {gt}
            }}
            {arcane}
            fn simd_ge(self, a: {nt}, b: {nt}) -> {nt} {{
                {ge}
            }}

            {arcane}
            fn blend(self, mask: {nt}, if_true: {nt}, if_false: {nt}) -> {nt} {{
                {blend_body}
            }}
    "#,
        eq = wrap_cmp(&format!("vceqq_{ns}")),
        ne = wrap_ne,
        lt = wrap_cmp(&format!("vcltq_{ns}")),
        le = wrap_cmp(&format!("vcleq_{ns}")),
        gt = wrap_cmp(&format!("vcgtq_{ns}")),
        ge = wrap_cmp(&format!("vcgeq_{ns}")),
    });

    // Reduce add
    if ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn reduce_add(self, a: {nt}) -> {elem} {{
                vaddvq_{ns}(a)
            }}
        "#});
    } else {
        // u64: vaddvq_u64 exists on NEON
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn reduce_add(self, a: {nt}) -> {elem} {{
                vaddvq_{ns}(a)
            }}
        "#});
    }

    // Bitwise
    body.push_str(&formatdoc! {r#"
            {arcane}
            fn not(self, a: {nt}) -> {nt} {{
                {not_body}
            }}
            {arcane}
            fn bitand(self, a: {nt}, b: {nt}) -> {nt} {{
                vandq_{ns}(a, b)
            }}
            {arcane}
            fn bitor(self, a: {nt}, b: {nt}) -> {nt} {{
                vorrq_{ns}(a, b)
            }}
            {arcane}
            fn bitxor(self, a: {nt}, b: {nt}) -> {nt} {{
                veorq_{ns}(a, b)
            }}
    "#});

    // Shifts
    let max_sh = ty.elem_bits - 1;
    body.push_str(&formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: {nt}) -> {nt} {{
                vshlq_n_{ns}::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {nt}) -> {nt} {{
                const {{ assert!(N >= 0 && N <= {max_sh}) }};
                {shr_logical}
            }}
    "#});

    if ty.signed {
        let us = &ns[1..]; // "8", "16"
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {nt}) -> {nt} {{
                const {{ assert!(N >= 0 && N <= {max_sh}) }};
                vshlq_{ns}(a, vdupq_n_s{us}((-N) as i{us}))
            }}
        "#});
    }

    // Uniform variable shifts. Same `vshlq_*` + splatted-count lowering the
    // const forms use; the count is clamped because USHL/SSHL read only the
    // low 8 bits of each shift-amount lane as a signed byte, so an unclamped
    // count of 256 would wrap to a no-op instead of the contracted zero.
    if ty.elem_bits == 16 {
        let eb = ty.elem_bits;
        let us = &ns[1..]; // "8", "16"
        let max_sh = eb - 1;
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn shl_uniform(self, a: {nt}, count: u32) -> {nt} {{
                vshlq_{ns}(a, vdupq_n_s{us}(count.min({eb}) as i{us}))
            }}
        "#});

        let shr_logical_uniform = if ty.signed {
            format!(
                "vreinterpretq_{ns}_u{us}(vshlq_u{us}(vreinterpretq_u{us}_{ns}(a), vdupq_n_s{us}(-(count.min({eb}) as i{us}))))"
            )
        } else {
            format!("vshlq_{ns}(a, vdupq_n_s{us}(-(count.min({eb}) as i{us})))")
        };
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn shr_logical_uniform(self, a: {nt}, count: u32) -> {nt} {{
                {shr_logical_uniform}
            }}
        "#});

        if ty.signed {
            body.push_str(&formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_uniform(self, a: {nt}, count: u32) -> {nt} {{
                // Clamping to lane_bits - 1 gives the contracted sign fill.
                vshlq_{ns}(a, vdupq_n_s{us}(-(count.min({max_sh}) as i{us})))
            }}
            "#});
        }
    }

    if ty.has_saturating() {
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn saturating_add(self, a: {nt}, b: {nt}) -> {nt} {{
                vqaddq_{ns}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: {nt}, b: {nt}) -> {nt} {{
                vqsubq_{ns}(a, b)
            }}
        "#});
    }

    // Boolean reductions
    if ty.elem_bits <= 16 {
        // Sign-bit contract: every lane's sign bit set <=> every lane is
        // negative when read as signed <=> the signed maximum is negative.
        // One reduction either way, and measured faster than the unsigned-min
        // form on M4 Pro.
        let ss = format!("s{}", ty.elem_bits);
        let to_s = format!("vreinterpretq_{ss}_{ns}");
        let all_true_body = if ty.signed {
            format!("vmaxvq_{ns}(a) < 0")
        } else {
            format!("vmaxvq_{ss}({to_s}(a)) < 0")
        };
        let any_true_body = if ty.signed {
            format!("vminvq_{ns}(a) < 0")
        } else {
            format!("vminvq_{ss}({to_s}(a)) < 0")
        };
        // Bitmask for 8-bit: extract high bit of each byte
        // Bitmask for 16-bit: extract high bit of each 16-bit lane
        let bitmask_body = generate_neon_bitmask(ty);
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn all_true(self, a: {nt}) -> bool {{
                {all_true_body}
            }}

            {arcane}
            fn any_true(self, a: {nt}) -> bool {{
                {any_true_body}
            }}

            {arcane}
            fn bitmask(self, a: {nt}) -> u32 {{
        {bitmask_body}
            }}
        "#});
    } else {
        // u64: 2 lanes, manual extraction (NEON lacks vminvq_u64/vmaxvq_u64)
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn all_true(self, a: {nt}) -> bool {{
                // NEON has no 64-bit horizontal reduction; AND the lanes and
                // test the sign bit once.
                ((vgetq_lane_u64::<0>(a) & vgetq_lane_u64::<1>(a)) >> 63) != 0
            }}

            {arcane}
            fn any_true(self, a: {nt}) -> bool {{
                ((vgetq_lane_u64::<0>(a) | vgetq_lane_u64::<1>(a)) >> 63) != 0
            }}

            {arcane}
            fn bitmask(self, a: {nt}) -> u32 {{
                let shift = vshrq_n_{ns}::<63>(a);
                let lane0 = vgetq_lane_{ns}::<0>(shift) as u32;
                let lane1 = vgetq_lane_{ns}::<1>(shift) as u32;
                lane0 | (lane1 << 1)
            }}
        "#});
    }

    body.push_str(&integer_methods);
    body.push_str("    }\n");
    body
}

/// NEON has no `movemask`. How best to pack one sign bit per lane into an
/// integer depends on the lane count — measured on an Apple M4 Pro, isolated
/// (`benchmarks/neon_bitmask_apple-m4-pro_2026-09-08.md`):
///
/// - **16 lanes (8-bit):** weight the lanes, then a *narrow* `vpadd_u8` tree.
///   A single `vaddv` over 16 bytes is the worst option measured (1.37x).
/// - **8 lanes (16-bit):** weight the lanes, then one `vaddvq_u16`.
/// - **4 lanes (32-bit):** shift each lane to 0/1, shift into position, then
///   one `vaddvq_u32`. Handled by the caller for the 32-bit types.
/// - **2 lanes (64-bit):** two `umov`s win; the vector setup does not pay for
///   itself. Handled separately.
///
/// `vcltzq_*` ("lane < 0") is what makes the AND-with-weights form correct:
/// it yields an all-ones lane, so `AND` selects the weight. An arithmetic
/// shift to 0/1 cannot be ANDed with the weights — `1 & 2 == 0`.
fn generate_neon_bitmask(ty: &IntVecType) -> String {
    let ns = ty.neon_suffix();
    let signed_suffix = format!("s{}", ty.elem_bits);
    let as_signed = if ty.signed {
        "a".to_string()
    } else {
        format!("vreinterpretq_{signed_suffix}_{ns}(a)")
    };
    match ty.elem_bits {
        8 => formatdoc! {r#"
            {{
                let powers: [u8; 16] = [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128];
                let pow_vec: uint8x16_t = crate::simd_storage::copy(&powers);
                let weighted = vandq_u8(vcltzq_s8({as_signed}), pow_vec);
                let s = vpadd_u8(vget_low_u8(weighted), vget_high_u8(weighted));
                let s = vpadd_u8(s, s);
                let s = vpadd_u8(s, s);
                u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(s)))
            }}"#},
        16 => formatdoc! {r#"
            {{
                let powers: [u16; 8] = [1,2,4,8,16,32,64,128];
                let pow_vec: uint16x8_t = crate::simd_storage::copy(&powers);
                u32::from(vaddvq_u16(vandq_u16(vcltzq_s16({as_signed}), pow_vec)))
            }}"#},
        _ => unreachable!("bitmask for 32/64-bit handled separately"),
    }
}

// ============================================================================
// WASM Implementation Generation
// ============================================================================

pub(super) fn generate_wasm_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.native_on_wasm() {
            code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
            code.push_str(&generate_wasm_native_int_impl(ty));
            code.push('\n');
        }
    }
    code
}

fn generate_wasm_native_int_impl(ty: &IntVecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "Wasm128Token");
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let wp = ty.wasm_prefix();

    // WASM intrinsic naming:
    // Arithmetic: i8x16_add (type-agnostic for add/sub)
    // Min/max: i8x16_min_s / u8x16_min_u (type-specific)
    // Comparisons: i8x16_eq / u8x16_gt (some are signed/unsigned)
    // Shifts: i8x16_shl (type-agnostic), i8x16_shr (signed), u8x16_shr (unsigned)

    let signed_wp = format!("{}x{}", ty.signed_elem(), ty.lanes_per_128());

    // For WASM, add/sub are the same regardless of signedness
    let arith_wp = &signed_wp; // i8x16, i16x8, i64x2

    let mut body = formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(self, v: {elem}) -> v128 {{
                {wp}_splat(v)
            }}

            #[inline(always)]
            fn zero(self) -> v128 {{
                {wp}_splat(0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ {arith_wp}_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ {arith_wp}_sub(a, b) }}
    "#};

    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ {arith_wp}_mul(a, b) }}
        "#});
    }

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ {arith_wp}_neg(a) }}
        "#});
    }

    // Min/max - WASM has both signed and unsigned variants
    let min_fn = if ty.signed {
        format!("{signed_wp}_min")
    } else {
        format!("{wp}_min")
    };
    let max_fn = if ty.signed {
        format!("{signed_wp}_max")
    } else {
        format!("{wp}_max")
    };

    if ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{ {min_fn}(a, b) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ {max_fn}(a, b) }}
        "#});
    } else {
        // u64: polyfill via compare + select
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{
                // u64 min polyfill: compare then select
                let mask = <Self as {trait_name}>::simd_lt(self, a, b);
                v128_bitselect(a, b, mask)
            }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{
                let mask = <Self as {trait_name}>::simd_gt(self, a, b);
                v128_bitselect(a, b, mask)
            }}
        "#});
    }

    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{ {signed_wp}_abs(a) }}
        "#});
    }

    // Comparisons
    let eq_fn = format!("{signed_wp}_eq");
    let ne_fn = format!("{signed_wp}_ne");
    let lt_fn = if ty.signed {
        format!("{signed_wp}_lt")
    } else {
        format!("{wp}_lt")
    };
    let le_fn = if ty.signed {
        format!("{signed_wp}_le")
    } else {
        format!("{wp}_le")
    };
    let gt_fn = if ty.signed {
        format!("{signed_wp}_gt")
    } else {
        format!("{wp}_gt")
    };
    let ge_fn = if ty.signed {
        format!("{signed_wp}_ge")
    } else {
        format!("{wp}_ge")
    };

    if ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ {eq_fn}(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ {ne_fn}(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ {lt_fn}(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ {le_fn}(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ {gt_fn}(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ {ge_fn}(a, b) }}
        "#});
    } else {
        // u64: only eq/ne are native on WASM, lt/gt/le/ge need bias trick
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ i64x2_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ i64x2_ne(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{
                // Unsigned comparison via bias trick
                let bias = i64x2_splat(i64::MIN);
                let sa = v128_xor(a, bias);
                let sb = v128_xor(b, bias);
                i64x2_gt(sa, sb)
            }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{
                <Self as {trait_name}>::simd_gt(self, b, a)
            }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(self, a, b))
            }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(self, b, a))
            }}
        "#});
    }

    // Blend
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}
    "#});

    // Reduce add
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn reduce_add(self, a: v128) -> {elem} {{
                let arr = <Self as {trait_name}>::to_array(self, a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}
    "#});

    // Bitwise
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
    "#});

    // Shifts
    let shl_fn = format!("{signed_wp}_shl");
    let shr_fn = if ty.signed {
        format!("{signed_wp}_shr")
    } else {
        format!("{wp}_shr")
    };

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: v128) -> v128 {{ {shl_fn}(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {{ {unsigned_shr}(a, N as u32) }}
    "#,
        unsigned_shr = if ty.signed {
            // Logical shift must use the u-prefixed intrinsic: the i-prefixed
            // `_shr` is the arithmetic (sign-filling) form in WASM naming.
            format!("u{}_shr", &signed_wp[1..])
        } else {
            shr_fn.clone()
        },
    });

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {{ {shr_fn}(a, N as u32) }}
        "#});
    }

    // Uniform variable shifts. WASM's shift intrinsics take the count modulo
    // the lane width ("Only the low bits of the shift amount are used if the
    // shift amount is greater than the lane width" -- core_arch/wasm32), which
    // is the one backend that disagrees with the portable contract. A splatted
    // all-ones/all-zero mask restores it; for the arithmetic form clamping the
    // count to lane_bits - 1 already gives the sign fill.
    if ty.elem_bits == 16 {
        let eb = ty.elem_bits;
        let max_sh = eb - 1;
        let selem = ty.signed_elem();
        let unsigned_shr = if ty.signed {
            format!("u{}_shr", &signed_wp[1..])
        } else {
            shr_fn.clone()
        };
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shl_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: {selem} = if count < {eb} {{ -1 }} else {{ 0 }};
                v128_and({shl_fn}(a, count), {signed_wp}_splat(keep))
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: {selem} = if count < {eb} {{ -1 }} else {{ 0 }};
                v128_and({unsigned_shr}(a, count), {signed_wp}_splat(keep))
            }}
        "#});

        if ty.signed {
            body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_uniform(self, a: v128, count: u32) -> v128 {{
                {shr_fn}(a, if count > {max_sh} {{ {max_sh} }} else {{ count }})
            }}
            "#});
        }
    }

    if ty.has_saturating() {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn saturating_add(self, a: v128, b: v128) -> v128 {{ {wp}_add_sat(a, b) }}

            #[inline(always)]
            fn saturating_sub(self, a: v128, b: v128) -> v128 {{ {wp}_sub_sat(a, b) }}
        "#});
    }

    // Boolean
    let bitmask_fn = format!("{signed_wp}_bitmask");
    // Sign-bit contract: the lane bitmask is exactly the set of sign bits.
    // `iNxM_all_true` tests nonzero, which is a different question.
    let full_mask = (1u64 << lanes) - 1;

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ {bitmask_fn}(a) as u64 == {full_mask} }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ {bitmask_fn}(a) != 0 }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ {bitmask_fn}(a) as u32 }}
        {integer_methods}
        }}
    "#});

    body
}

// ============================================================================
// Conversion Implementations
// ============================================================================

/// Generate x86 conversion impls for the new types.
pub(super) fn generate_x86_additional_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I8x16Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_i8_to_u8(self, a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(self, a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I8x32Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_i8_to_u8(self, a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(self, a: __m256i) -> __m256i {{ a }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I16x8Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_i16_to_u16(self, a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(self, a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I16x16Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_i16_to_u16(self, a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(self, a: __m256i) -> __m256i {{ a }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl U64x2Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_u64_to_i64(self, a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(self, a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl U64x4Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_u64_to_i64(self, a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(self, a: __m256i) -> __m256i {{ a }}
        }}
    "#}
}

/// Generate NEON conversion impls for the native 128-bit integer bitcast types.
pub(super) fn generate_neon_additional_convert_impls() -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        impl I8x16Bitcast for archmage::NeonToken {{
            {arcane}
            fn bitcast_i8_to_u8(self, a: int8x16_t) -> uint8x16_t {{
                vreinterpretq_u8_s8(a)
            }}
            {arcane}
            fn bitcast_u8_to_i8(self, a: uint8x16_t) -> int8x16_t {{
                vreinterpretq_s8_u8(a)
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I16x8Bitcast for archmage::NeonToken {{
            {arcane}
            fn bitcast_i16_to_u16(self, a: int16x8_t) -> uint16x8_t {{
                vreinterpretq_u16_s16(a)
            }}
            {arcane}
            fn bitcast_u16_to_i16(self, a: uint16x8_t) -> int16x8_t {{
                vreinterpretq_s16_u16(a)
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U64x2Bitcast for archmage::NeonToken {{
            {arcane}
            fn bitcast_u64_to_i64(self, a: uint64x2_t) -> int64x2_t {{
                vreinterpretq_s64_u64(a)
            }}
            {arcane}
            fn bitcast_i64_to_u64(self, a: int64x2_t) -> uint64x2_t {{
                vreinterpretq_u64_s64(a)
            }}
        }}
    "#}
}

/// Generate WASM conversion impls for the native 128-bit integer bitcast types.
pub(super) fn generate_wasm_additional_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "wasm32")]
        impl I8x16Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i8_to_u8(self, a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(self, a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I16x8Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i16_to_u16(self, a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(self, a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U64x2Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u64_to_i64(self, a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(self, a: v128) -> v128 {{ a }}
        }}
    "#}
}

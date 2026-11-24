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

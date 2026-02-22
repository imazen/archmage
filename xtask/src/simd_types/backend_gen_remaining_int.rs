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
        let native = self.neon_native_type();
        if self.native_on_neon() {
            native
        } else {
            let count = self.width_bits / 128;
            format!("[{native}; {count}]")
        }
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

    /// NEON reinterpret from unsigned to this type's repr
    fn neon_reinterpret_from_u(&self) -> String {
        let ns = self.neon_suffix();
        let us = match self.elem_bits {
            8 => "u8",
            16 => "u16",
            64 => "u64",
            _ => unreachable!(),
        };
        if self.signed {
            format!("vreinterpretq_{ns}_{us}")
        } else {
            String::new() // Already unsigned, no reinterpret needed
        }
    }

    /// NEON reinterpret from this type to unsigned
    fn neon_reinterpret_to_u(&self) -> String {
        let ns = self.neon_suffix();
        let us = match self.elem_bits {
            8 => "u8",
            16 => "u16",
            64 => "u64",
            _ => unreachable!(),
        };
        if self.signed {
            format!("vreinterpretq_{us}_{ns}")
        } else {
            String::new() // Already unsigned
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

    /// WASM intrinsic prefix for this element type: "i8x16", "u8x16", "i16x8", etc.
    fn wasm_prefix(&self) -> String {
        format!("{}x{}", self.elem, self.lanes_per_128())
    }

    /// Lanes in a 128-bit sub-vector
    fn lanes_per_128(&self) -> usize {
        128 / self.elem_bits
    }

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
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

    /// Bias value for unsigned comparison (as signed literal)
    fn unsigned_bias_literal(&self) -> &'static str {
        match self.elem_bits {
            8 => "0x80u8 as i8",
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
// Backend Trait Generation
// ============================================================================

pub(super) fn generate_int_backend_trait(ty: &IntVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();

    let mut methods = String::new();

    // Construction
    methods.push_str(&formatdoc! {r#"
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

            /// Lane-wise addition (wrapping).
            fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction (wrapping).
            fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;
    "#});

    // Mul only for i16/u16
    if ty.has_native_mul() {
        methods.push_str(&formatdoc! {r#"
            /// Lane-wise multiplication (low {elem_bits} bits of product).
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;
        "#, elem_bits = ty.elem_bits});
    }

    // Neg only for signed types
    if ty.signed {
        methods.push_str(&formatdoc! {r#"
            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;
        "#});
    }

    // Math
    methods.push_str(&formatdoc! {r#"

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;
    "#});

    // Abs only for signed types
    if ty.signed {
        methods.push_str(&formatdoc! {r#"
            /// Lane-wise absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;
        "#});
    }

    // Comparisons, reductions, bitwise, shifts, boolean
    methods.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

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

            /// Sum all {lanes} lanes (wrapping).
            fn reduce_add(a: Self::Repr) -> {elem};

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

            /// Logical shift right by constant (zero-filling).
            fn shr_logical_const<const N: i32>(a: Self::Repr) -> Self::Repr;
    "#});

    // Arithmetic shift right only for signed types
    if ty.signed {
        methods.push_str(&formatdoc! {r#"
            /// Arithmetic shift right by constant (sign-extending).
            fn shr_arithmetic_const<const N: i32>(a: Self::Repr) -> Self::Repr;
        "#});
    }

    // Boolean
    methods.push_str(&formatdoc! {r#"

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set.
            fn any_true(a: Self::Repr) -> bool;

            /// Extract the high bit of each lane as a bitmask.
            fn bitmask(a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
            }}
    "#});

    let name = ty.name();
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
        {methods}
        }}
    "#}
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

pub(super) fn generate_x86_int_impls(
    types: &[IntVecType],
    token: &str,
    max_width: usize,
) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("\n#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_int_impl(ty, token));
        code.push('\n');
    }
    code
}

fn generate_x86_int_impl(ty: &IntVecType, token: &str) -> String {
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;
    let elem = ty.elem;
    let set1_suf = ty.x86_set1_suffix();
    let arith_suf = ty.x86_arith_suffix();
    let set1_cast = ty.x86_set1_cast();

    let mut body = String::new();

    // Construction
    body.push_str(&formatdoc! {r#"
            type Repr = {inner};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {inner} {{
                unsafe {{ {p}_set1_{set1_suf}(v{set1_cast}) }}
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
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_{arith_suf}(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{arith_suf}(a, b) }}
            }}
    "#});

    // Mul
    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mullo_epi16(a, b) }}
            }}
        "#});
    }

    // Neg
    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{arith_suf}({p}_setzero_si{bits}(), a) }}
            }}
        "#});
    }

    // Math: min/max
    if ty.has_native_minmax() {
        let mm_suf = ty.x86_minmax_suffix();
        body.push_str(&formatdoc! {r#"

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_{mm_suf}(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_{mm_suf}(a, b) }}
            }}
        "#});
    } else {
        // u64: polyfill via bias trick + signed compare + blend
        body.push_str(&formatdoc! {r#"

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let bias = {p}_set1_epi64x(i64::MIN);
                    let a_biased = {p}_xor_si{bits}(a, bias);
                    let b_biased = {p}_xor_si{bits}(b, bias);
                    let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                    {p}_blendv_epi8(a, b, mask)
                }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let bias = {p}_set1_epi64x(i64::MIN);
                    let a_biased = {p}_xor_si{bits}(a, bias);
                    let b_biased = {p}_xor_si{bits}(b, bias);
                    let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                    {p}_blendv_epi8(b, a, mask)
                }}
            }}
        "#});
    }

    // Abs
    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                unsafe {{ {p}_abs_{arith_suf}(a) }}
            }}
        "#});
    }

    // Comparisons
    let cmp_suf = ty.x86_arith_suffix(); // eq/gt intrinsics use same suffix
    if ty.signed {
        // Signed: direct cmpgt
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_{cmp_suf}(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_{cmp_suf}(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_{set1_suf}(-1))
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_{cmp_suf}(b, a) }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = {p}_cmpgt_{cmp_suf}(a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_{set1_suf}(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_{cmp_suf}(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = {p}_cmpgt_{cmp_suf}(b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_{set1_suf}(-1))
                }}
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

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_{cmp_suf}(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_{cmp_suf}(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_{set1_suf}(-1{set1_cast}))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let bias = {bias_set1};
                    let sa = {p}_xor_si{bits}(a, bias);
                    let sb = {p}_xor_si{bits}(b, bias);
                    {p}_cmpgt_{cmp_suf}(sa, sb)
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(b, a)
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = <Self as {trait_name}>::simd_gt(a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_{set1_suf}(-1{set1_cast}))
                }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = <Self as {trait_name}>::simd_gt(b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_{set1_suf}(-1{set1_cast}))
                }}
            }}
        "#});
    }

    // Blend
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}
    "#});

    // Reduce add
    let reduce_body = generate_x86_int_reduce_add(ty);
    body.push_str(&formatdoc! {r#"

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> {elem} {{
        {reduce_body}
            }}
    "#});

    // Bitwise
    body.push_str(&formatdoc! {r#"

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{ {p}_andnot_si{bits}(a, {p}_set1_{set1_suf}(-1{set1_cast})) }}
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
    "#});

    // Shifts
    body.push_str(&generate_x86_int_shifts(ty));

    // Boolean
    body.push_str(&generate_x86_int_boolean(ty));

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
        {body}
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
                let arr = <Self as {trait_name}>::to_array(a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)"}
}

fn generate_x86_int_shifts(ty: &IntVecType) -> String {
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;

    if ty.elem_bits == 8 {
        // 8-bit: polyfill via 16-bit shift + mask
        let mut code = formatdoc! {r#"

            // ====== Shifts (polyfill via 16-bit) ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{
                    let shifted = {p}_slli_epi16::<N>(a);
                    let mask = {p}_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
                    {p}_and_si{bits}(shifted, mask)
                }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{
                    let shifted = {p}_srli_epi16::<N>(a);
                    let mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                    {p}_and_si{bits}(shifted, mask)
                }}
            }}
        "#};

        if ty.signed {
            code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{
                    let shifted = {p}_srli_epi16::<N>(a);
                    let byte_mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                    let logical = {p}_and_si{bits}(shifted, byte_mask);
                    let zero = {p}_setzero_si{bits}();
                    let sign = {p}_cmpgt_epi8(zero, a);
                    let fill = {p}_set1_epi8((0xFFu8.wrapping_shl(8u32.wrapping_sub(N as u32))) as i8);
                    {p}_or_si{bits}(logical, {p}_and_si{bits}(sign, fill))
                }}
            }}
            "#});
        }
        code
    } else if ty.elem_bits == 16 {
        // 16-bit: native shifts
        let mut code = formatdoc! {r#"

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi16::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi16::<N>(a) }}
            }}
        "#};

        if ty.signed {
            code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srai_epi16::<N>(a) }}
            }}
            "#});
        }
        code
    } else {
        // 64-bit: native shifts (same as i64)
        formatdoc! {r#"

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi64::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi64::<N>(a) }}
            }}
        "#}
    }
}

fn generate_x86_int_boolean(ty: &IntVecType) -> String {
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

            #[inline(always)]
            fn all_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_epi8(a) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_epi8(a) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {inner}) -> u32 {{
                unsafe {{ {p}_movemask_epi8(a) as u32 }}
            }}
            "#}
        }
        16 => {
            // 16-bit: all_true/any_true can use byte-level movemask (all bytes set = all 16-bit lanes set)
            // bitmask needs to extract one bit per 16-bit lane
            let all_mask_bytes = if ty.width_bits == 128 {
                "0xFFFF_u32 as i32".to_string()
            } else {
                "-1_i32".to_string()
            };
            let bitmask_mask = if ty.width_bits == 128 {
                "0xFF"
            } else {
                "0xFFFF"
            };
            formatdoc! {r#"

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_epi8(a) == {all_mask_bytes} }}
            }}

            #[inline(always)]
            fn any_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_epi8(a) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {inner}) -> u32 {{
                unsafe {{
                    let shifted = {p}_srai_epi16::<15>(a);
                    let packed = {p}_packs_epi16(shifted, shifted);
                    ({p}_movemask_epi8(packed) & {bitmask_mask}) as u32
                }}
            }}
            "#}
        }
        64 => {
            // 64-bit: use movemask_pd
            let all_mask = if ty.width_bits == 128 { "0x3" } else { "0xF" };
            formatdoc! {r#"

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
            "#}
        }
        _ => unreachable!(),
    }
}

// ============================================================================
// Scalar Implementation Generation
// ============================================================================

pub(super) fn generate_scalar_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str(&generate_scalar_int_impl(ty));
        code.push('\n');
    }
    code
}

fn generate_scalar_int_impl(ty: &IntVecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;

    let binary_wrapping = |method: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].{method}(b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes = || -> String {
        let cmp = if ty.signed { "<" } else { "<" };
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] {cmp} b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes = || -> String {
        let cmp = if ty.signed { ">" } else { ">" };
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] {cmp} b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let mask_val = |val: &str| -> String {
        // All-1s mask for this element type
        match ty.elem {
            "i8" => format!("-1{val}"),
            "u8" => format!("0xFF{val}"),
            "i16" => format!("-1{val}"),
            "u16" => format!("0xFFFF{val}"),
            "u64" => format!("u64::MAX{val}"),
            _ => unreachable!(),
        }
    };

    let cmp_op = |op: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| {
                let true_val = mask_val("");
                format!("if a[{i}] {op} b[{i}] {{ {true_val} }} else {{ 0 }}")
            })
            .collect();
        format!("[{}]", items.join(", "))
    };

    let blend_items: Vec<String> = (0..lanes)
        .map(|i| format!("if mask[{i}] != 0 {{ if_true[{i}] }} else {{ if_false[{i}] }}"))
        .collect();
    let blend_body = format!("[{}]", blend_items.join(", "));

    let all_true_check: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
    let all_true_body = all_true_check.join(" && ");

    let any_true_check: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
    let any_true_body = any_true_check.join(" || ");

    let bitmask_items: Vec<String> = (0..lanes)
        .map(|i| {
            let shift_by = match ty.elem_bits {
                8 | 16 | 64 => ty.elem_bits - 1,
                _ => unreachable!(),
            };
            format!("((a[{i}] >> {shift_by}) as u32 & 1) << {i}")
        })
        .collect();
    let bitmask_body = bitmask_items.join(" | ");

    let shl_items: Vec<String> = (0..lanes)
        .map(|i| format!("a[{i}].wrapping_shl(N as u32)"))
        .collect();
    let shl_body = format!("[{}]", shl_items.join(", "));

    let shr_logical_items: Vec<String> = (0..lanes)
        .map(|i| {
            if ty.signed {
                // For signed types, logical shift right needs cast to unsigned
                let unsigned = match ty.elem_bits {
                    8 => "u8",
                    16 => "u16",
                    _ => unreachable!(),
                };
                format!("(a[{i}] as {unsigned}).wrapping_shr(N as u32) as {elem}")
            } else {
                format!("a[{i}].wrapping_shr(N as u32)")
            }
        })
        .collect();
    let shr_logical_body = format!("[{}]", shr_logical_items.join(", "));

    let mut body = formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            #[inline(always)]
            fn splat(v: {elem}) -> {array} {{ [{splat_items}] }}

            #[inline(always)]
            fn zero() -> {array} {{ [0{elem}; {lanes}] }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{ *data }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{ arr }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{ *out = repr; }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{ repr }}

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{ {add} }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{ {sub} }}
    "#,
        splat_items = (0..lanes).map(|_| "v").collect::<Vec<_>>().join(", "),
        add = binary_wrapping("wrapping_add"),
        sub = binary_wrapping("wrapping_sub"),
    };

    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{ {mul} }}
        "#,
            mul = binary_wrapping("wrapping_mul"),
        });
    }

    if ty.signed {
        let neg_items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_neg()"))
            .collect();
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn neg(a: {array}) -> {array} {{ [{neg}] }}
        "#, neg = neg_items.join(", ")});
    }

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{ {min} }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{ {max} }}
    "#,
        min = min_lanes(),
        max = max_lanes(),
    });

    if ty.signed {
        let abs_items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_abs()"))
            .collect();
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{ [{abs}] }}
        "#, abs = abs_items.join(", ")});
    }

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{ {eq} }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{ {ne} }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{ {lt} }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{ {le} }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{ {gt} }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{ {ge} }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                {blend_body}
            }}

            #[inline(always)]
            fn reduce_add(a: {array}) -> {elem} {{
                a.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}

            #[inline(always)]
            fn not(a: {array}) -> {array} {{ [{not}] }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{ [{and}] }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{ [{or}] }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{ [{xor}] }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {array}) -> {array} {{ {shl_body} }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {array}) -> {array} {{ {shr_logical_body} }}
    "#,
        eq = cmp_op("=="),
        ne = cmp_op("!="),
        lt = cmp_op("<"),
        le = cmp_op("<="),
        gt = cmp_op(">"),
        ge = cmp_op(">="),
        not = (0..lanes).map(|i| format!("!a[{i}]")).collect::<Vec<_>>().join(", "),
        and = (0..lanes).map(|i| format!("a[{i}] & b[{i}]")).collect::<Vec<_>>().join(", "),
        or = (0..lanes).map(|i| format!("a[{i}] | b[{i}]")).collect::<Vec<_>>().join(", "),
        xor = (0..lanes).map(|i| format!("a[{i}] ^ b[{i}]")).collect::<Vec<_>>().join(", "),
    });

    if ty.signed {
        let shr_arith_items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_shr(N as u32)"))
            .collect();
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {array}) -> {array} {{
                [{shr_arith}]
            }}
        "#, shr_arith = shr_arith_items.join(", ")});
    }

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(a: {array}) -> bool {{ {all_true_body} }}

            #[inline(always)]
            fn any_true(a: {array}) -> bool {{ {any_true_body} }}

            #[inline(always)]
            fn bitmask(a: {array}) -> u32 {{ {bitmask_body} }}
        }}
    "#});

    body
}

// ============================================================================
// NEON Implementation Generation
// ============================================================================

pub(super) fn generate_neon_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
        if ty.native_on_neon() {
            code.push_str(&generate_neon_native_int_impl(ty));
        } else {
            code.push_str(&generate_neon_polyfill_int_impl(ty));
        }
        code.push('\n');
    }
    code
}

fn generate_neon_native_int_impl(ty: &IntVecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let ns = ty.neon_suffix();
    let nt = ty.neon_native_type();
    let _ut = ty.neon_unsigned_type();

    // Reinterpret helpers
    let from_u = if ty.signed {
        format!("vreinterpretq_{ns}_{}", &ns[1..]) // s8_u8, s16_u16
    } else {
        String::new()
    };
    let to_u = if ty.signed {
        format!("vreinterpretq_{}_{ns}", &ns[1..]) // u8_s8, u16_s16
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
        format!("{from_u}(vmvnq_{}(vceqq_{ns}(a, b)))", &ns[1..])
    } else {
        format!("vmvnq_{ns}(vceqq_{ns}(a, b))")
    };

    let blend_body = if ty.signed {
        format!("vbslq_{ns}({to_u}(mask), if_true, if_false)")
    } else {
        format!("vbslq_{ns}(mask, if_true, if_false)")
    };

    let not_body = if ty.signed {
        format!("vmvnq_{ns}(a)")
    } else {
        format!("vmvnq_{ns}(a)")
    };

    // For u64, there's no vmvnq_u64, need different approach
    let not_body = if ty.elem_bits == 64 {
        format!("veorq_{ns}(a, vdupq_n_{ns}(u64::MAX))")
    } else {
        not_body
    };

    let shr_logical = if ty.signed {
        // Signed logical shift right: cast to unsigned, shift, cast back
        let us = &ns[1..]; // "8", "16"
        format!("vreinterpretq_{ns}_u{us}(vshrq_n_u{us}::<N>(vreinterpretq_u{us}_{ns}(a)))")
    } else if ty.elem_bits == 64 {
        format!("vshrq_n_{ns}::<N>(a)")
    } else {
        format!("vshrq_n_{ns}::<N>(a)")
    };

    let mut body = formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {nt};

            #[inline(always)]
            fn splat(v: {elem}) -> {nt} {{
                unsafe {{ vdupq_n_{ns}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {nt} {{
                unsafe {{ vdupq_n_{ns}(0) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {nt} {{
                unsafe {{ vld1q_{ns}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {nt} {{
                unsafe {{ vld1q_{ns}(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(repr: {nt}, out: &mut {array}) {{
                unsafe {{ vst1q_{ns}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {nt}) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                unsafe {{ vst1q_{ns}(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: {nt}, b: {nt}) -> {nt} {{ unsafe {{ vaddq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sub(a: {nt}, b: {nt}) -> {nt} {{ unsafe {{ vsubq_{ns}(a, b) }} }}
    "#};

    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(a: {nt}, b: {nt}) -> {nt} {{ unsafe {{ vmulq_{ns}(a, b) }} }}
        "#});
    }

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(a: {nt}) -> {nt} {{ unsafe {{ vnegq_{ns}(a) }} }}
        "#});
    }

    body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(a: {nt}, b: {nt}) -> {nt} {{ unsafe {{ vminq_{ns}(a, b) }} }}
            #[inline(always)]
            fn max(a: {nt}, b: {nt}) -> {nt} {{ unsafe {{ vmaxq_{ns}(a, b) }} }}
    "#});

    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(a: {nt}) -> {nt} {{ unsafe {{ vabsq_{ns}(a) }} }}
        "#});
    }

    // Comparisons
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {eq} }}
            }}
            #[inline(always)]
            fn simd_ne(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {ne} }}
            }}
            #[inline(always)]
            fn simd_lt(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {lt} }}
            }}
            #[inline(always)]
            fn simd_le(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {le} }}
            }}
            #[inline(always)]
            fn simd_gt(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {gt} }}
            }}
            #[inline(always)]
            fn simd_ge(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ {ge} }}
            }}

            #[inline(always)]
            fn blend(mask: {nt}, if_true: {nt}, if_false: {nt}) -> {nt} {{
                unsafe {{ {blend_body} }}
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
            #[inline(always)]
            fn reduce_add(a: {nt}) -> {elem} {{
                unsafe {{ vaddvq_{ns}(a) }}
            }}
        "#});
    } else {
        // u64: vaddvq_u64 exists on NEON
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn reduce_add(a: {nt}) -> {elem} {{
                unsafe {{ vaddvq_{ns}(a) }}
            }}
        "#});
    }

    // Bitwise
    body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn not(a: {nt}) -> {nt} {{
                unsafe {{ {not_body} }}
            }}
            #[inline(always)]
            fn bitand(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ vandq_{ns}(a, b) }}
            }}
            #[inline(always)]
            fn bitor(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ vorrq_{ns}(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(a: {nt}, b: {nt}) -> {nt} {{
                unsafe {{ veorq_{ns}(a, b) }}
            }}
    "#});

    // Shifts
    body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shl_const<const N: i32>(a: {nt}) -> {nt} {{
                unsafe {{ vshlq_n_{ns}::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {nt}) -> {nt} {{
                unsafe {{ {shr_logical} }}
            }}
    "#});

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {nt}) -> {nt} {{
                unsafe {{ vshrq_n_{ns}::<N>(a) }}
            }}
        "#});
    }

    // Boolean reductions
    if ty.elem_bits <= 16 {
        // 8-bit and 16-bit: use vminvq/vmaxvq on unsigned interpretation
        let all_true_body = if ty.signed {
            format!("vminvq_{}({to_u}(a)) != 0", &ns[1..])
        } else {
            format!("vminvq_{ns}(a) != 0")
        };
        let any_true_body = if ty.signed {
            format!("vmaxvq_{}({to_u}(a)) != 0", &ns[1..])
        } else {
            format!("vmaxvq_{ns}(a) != 0")
        };
        // Bitmask for 8-bit: extract high bit of each byte
        // Bitmask for 16-bit: extract high bit of each 16-bit lane
        let bitmask_body = generate_neon_bitmask(ty);
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(a: {nt}) -> bool {{
                unsafe {{ {all_true_body} }}
            }}

            #[inline(always)]
            fn any_true(a: {nt}) -> bool {{
                unsafe {{ {any_true_body} }}
            }}

            #[inline(always)]
            fn bitmask(a: {nt}) -> u32 {{
        {bitmask_body}
            }}
        "#});
    } else {
        // u64: 2 lanes, manual extraction
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(a: {nt}) -> bool {{
                unsafe {{ vminvq_{ns}(a) != 0 }}
            }}

            #[inline(always)]
            fn any_true(a: {nt}) -> bool {{
                unsafe {{ vmaxvq_{ns}(a) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {nt}) -> u32 {{
                unsafe {{
                    let shift = vshrq_n_{ns}::<63>(a);
                    let lane0 = vgetq_lane_{ns}::<0>(shift) as u32;
                    let lane1 = vgetq_lane_{ns}::<1>(shift) as u32;
                    lane0 | (lane1 << 1)
                }}
            }}
        "#});
    }

    body.push_str("    }\n");
    body
}

fn generate_neon_bitmask(ty: &IntVecType) -> String {
    let ns = ty.neon_suffix();
    match ty.elem_bits {
        8 => {
            // Extract high bit of each byte.
            // Use vshrq_n to get sign bit, then narrow+compress.
            // For NEON, we use a scalar-ish approach: shift right by 7, then
            // pack down and extract.
            formatdoc! {r#"
                unsafe {{
                    // Shift each byte right by 7 to isolate sign bit
                    let bits = vshrq_n_{ns}::<7>(a);
                    // Use polynomial evaluation to pack bits
                    // Each byte is now 0 or 1, multiply by position powers of 2
                    let powers: [u8; 16] = [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128];
                    let pow_vec = vld1q_u8(powers.as_ptr());
                    let weighted = vmulq_u8({reinterpret}bits{rparen}, pow_vec);
                    // Sum pairs: add adjacent bytes
                    let pair_sum = vpaddlq_u8(weighted);
                    let quad_sum = vpaddlq_u16(pair_sum);
                    let oct_sum = vpaddlq_u32(quad_sum);
                    // Extract low and high byte
                    let lo = vgetq_lane_u64::<0>(oct_sum) as u32;
                    let hi = vgetq_lane_u64::<1>(oct_sum) as u32;
                    lo | (hi << 8)
                }}"#,
                reinterpret = if ty.signed { format!("vreinterpretq_u8_{ns}(") } else { String::new() },
                rparen = if ty.signed { ")" } else { "" },
            }
        }
        16 => {
            let lanes = ty.lanes_per_128();
            let items: Vec<String> = (0..lanes)
                .map(|i| {
                    if ty.signed {
                        let us = &ns[1..]; // "16"
                        format!("(vgetq_lane_u{us}::<{i}>(vreinterpretq_u{us}_{ns}(vshrq_n_{ns}::<15>(a))) as u32 & 1) << {i}")
                    } else {
                        format!("(vgetq_lane_{ns}::<{i}>(vshrq_n_{ns}::<15>(a)) as u32 & 1) << {i}")
                    }
                })
                .collect();
            formatdoc! {r#"
                unsafe {{
                    {bitmask}
                }}"#,
                bitmask = items.join(" | "),
            }
        }
        _ => unreachable!("bitmask for 64-bit handled separately"),
    }
}

fn generate_neon_polyfill_int_impl(ty: &IntVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let ns = ty.neon_suffix();
    let _nt = ty.neon_native_type();
    let sub_count = ty.sub_count();
    let lanes_per_128 = ty.lanes_per_128();

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

    let from_u_fn = ty.neon_reinterpret_from_u();
    let to_u_fn = ty.neon_reinterpret_to_u();

    let cmp_op = |intrinsic: &str| -> String {
        if ty.signed {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("{from_u_fn}({intrinsic}(a[{i}], b[{i}]))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        } else {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        }
    };

    let ne_op = || -> String {
        let us = match ty.elem_bits {
            8 => "u8",
            16 => "u16",
            64 => "u64",
            _ => unreachable!(),
        };
        if ty.signed {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("{from_u_fn}(vmvnq_{us}(vceqq_{ns}(a[{i}], b[{i}])))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        } else {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vmvnq_{us}(vceqq_{ns}(a[{i}], b[{i}]))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        }
    };

    let v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", ");
    let z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", ");

    let load_lanes: Vec<String> = (0..sub_count)
        .map(|i| {
            let offset = i * lanes_per_128;
            format!("vld1q_{ns}(data.as_ptr().add({offset}))")
        })
        .collect();
    let load_body = load_lanes.join(", ");

    let store_lines: Vec<String> = (0..sub_count)
        .map(|i| {
            let offset = i * lanes_per_128;
            format!("vst1q_{ns}(out.as_mut_ptr().add({offset}), repr[{i}]);")
        })
        .collect();
    let store_body = store_lines.join("\n                    ");

    let blend_items: Vec<String> = (0..sub_count)
        .map(|i| {
            if ty.signed {
                format!("vbslq_{ns}({to_u_fn}(mask[{i}]), if_true[{i}], if_false[{i}])")
            } else {
                format!("vbslq_{ns}(mask[{i}], if_true[{i}], if_false[{i}])")
            }
        })
        .collect();
    let blend_body = format!("unsafe {{ [{}] }}", blend_items.join(", "));

    let not_op = if ty.elem_bits == 64 {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("veorq_{ns}(a[{i}], vdupq_n_{ns}(u64::MAX))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    } else {
        unary_op(&format!("vmvnq_{ns}"))
    };

    let shl_items: Vec<String> = (0..sub_count)
        .map(|i| format!("vshlq_n_{ns}::<N>(a[{i}])"))
        .collect();
    let shl_body = format!("unsafe {{ [{}] }}", shl_items.join(", "));

    let shr_logical_items: Vec<String> = (0..sub_count)
        .map(|i| {
            if ty.signed {
                let us = &ns[1..];
                format!(
                    "vreinterpretq_{ns}_u{us}(vshrq_n_u{us}::<N>(vreinterpretq_u{us}_{ns}(a[{i}])))"
                )
            } else {
                format!("vshrq_n_{ns}::<N>(a[{i}])")
            }
        })
        .collect();
    let shr_logical_body = format!("unsafe {{ [{}] }}", shr_logical_items.join(", "));

    let mut code = formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

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
                    let z = vdupq_n_{ns}(0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{ [{load_body}] }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_body}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
    "#,
        add = binary_op(&format!("vaddq_{ns}")),
        sub = binary_op(&format!("vsubq_{ns}")),
    };

    if ty.has_native_mul() {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
        "#, mul = binary_op(&format!("vmulq_{ns}"))});
    }

    if ty.signed {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
        "#, neg = unary_op(&format!("vnegq_{ns}"))});
    }

    code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
    "#,
        min = binary_op(&format!("vminq_{ns}")),
        max = binary_op(&format!("vmaxq_{ns}")),
    });

    if ty.signed && ty.elem_bits <= 16 {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}
        "#, abs = unary_op(&format!("vabsq_{ns}"))});
    }

    code.push_str(&formatdoc! {r#"

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
                {blend_body}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                let mut sum = 0{elem};
                for i in 0..{sub_count} {{
                    sum = sum.wrapping_add(unsafe {{ vaddvq_{ns}(a[i]) }});
                }}
                sum
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not_op} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{ {shl_body} }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{ {shr_logical_body} }}
    "#,
        eq = cmp_op(&format!("vceqq_{ns}")),
        ne = ne_op(),
        lt = cmp_op(&format!("vcltq_{ns}")),
        le = cmp_op(&format!("vcleq_{ns}")),
        gt = cmp_op(&format!("vcgtq_{ns}")),
        ge = cmp_op(&format!("vcgeq_{ns}")),
        bitand = binary_op(&format!("vandq_{ns}")),
        bitor = binary_op(&format!("vorrq_{ns}")),
        bitxor = binary_op(&format!("veorq_{ns}")),
    });

    if ty.signed {
        let shr_arith_items: Vec<String> = (0..sub_count)
            .map(|i| format!("vshrq_n_{ns}::<N>(a[{i}])"))
            .collect();
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                unsafe {{ [{shr_arith}] }}
            }}
        "#, shr_arith = shr_arith_items.join(", ")});
    }

    // Boolean: delegate to sub-vector operations
    let all_true_items: Vec<String> = (0..sub_count)
        .map(|i| {
            if ty.signed {
                let us = &ns[1..];
                format!("vminvq_u{us}(vreinterpretq_u{us}_{ns}(a[{i}])) != 0")
            } else {
                format!("vminvq_{ns}(a[{i}]) != 0")
            }
        })
        .collect();
    let any_true_items: Vec<String> = (0..sub_count)
        .map(|i| {
            if ty.signed {
                let us = &ns[1..];
                format!("vmaxvq_u{us}(vreinterpretq_u{us}_{ns}(a[{i}])) != 0")
            } else {
                format!("vmaxvq_{ns}(a[{i}]) != 0")
            }
        })
        .collect();

    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                unsafe {{ {all_true} }}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                unsafe {{ {any_true} }}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
                // Delegate to NeonToken native bitmask per sub-vector, combine
                let mut result = 0u32;
                for i in 0..{sub_count} {{
                    result |= (<archmage::NeonToken as {native_trait}>::bitmask(a[i])) << (i * {lanes_per_128});
                }}
                result
            }}
        }}
    "#,
        all_true = all_true_items.join(" && "),
        any_true = any_true_items.join(" || "),
        native_trait = {
            let upper = match ty.elem {
                "i8" => "I8",
                "u8" => "U8",
                "i16" => "I16",
                "u16" => "U16",
                "u64" => "U64",
                _ => unreachable!(),
            };
            format!("{upper}x{}Backend", lanes_per_128)
        },
    });

    code
}

// ============================================================================
// WASM Implementation Generation
// ============================================================================

pub(super) fn generate_wasm_int_impls(types: &[IntVecType]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
        if ty.native_on_wasm() {
            code.push_str(&generate_wasm_native_int_impl(ty));
        } else {
            code.push_str(&generate_wasm_polyfill_int_impl(ty));
        }
        code.push('\n');
    }
    code
}

fn generate_wasm_native_int_impl(ty: &IntVecType) -> String {
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
            fn splat(v: {elem}) -> v128 {{
                {wp}_splat(v)
            }}

            #[inline(always)]
            fn zero() -> v128 {{
                {wp}_splat(0)
            }}

            #[inline(always)]
            fn load(data: &{array}) -> v128 {{
                unsafe {{ v128_load(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{
                unsafe {{ v128_load(arr.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ {arith_wp}_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ {arith_wp}_sub(a, b) }}
    "#};

    if ty.has_native_mul() {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(a: v128, b: v128) -> v128 {{ {arith_wp}_mul(a, b) }}
        "#});
    }

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(a: v128) -> v128 {{ {arith_wp}_neg(a) }}
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
            fn min(a: v128, b: v128) -> v128 {{ {min_fn}(a, b) }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{ {max_fn}(a, b) }}
        "#});
    } else {
        // u64: polyfill via compare + select
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{
                // u64 min polyfill: compare then select
                let mask = <Self as {trait_name}>::simd_lt(a, b);
                v128_bitselect(a, b, mask)
            }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{
                let mask = <Self as {trait_name}>::simd_gt(a, b);
                v128_bitselect(a, b, mask)
            }}
        "#});
    }

    if ty.signed && ty.elem_bits <= 16 {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(a: v128) -> v128 {{ {signed_wp}_abs(a) }}
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
            fn simd_eq(a: v128, b: v128) -> v128 {{ {eq_fn}(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ {ne_fn}(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ {lt_fn}(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ {le_fn}(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ {gt_fn}(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ {ge_fn}(a, b) }}
        "#});
    } else {
        // u64: only eq/ne are native on WASM, lt/gt/le/ge need bias trick
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ i64x2_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ i64x2_ne(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{
                // Unsigned comparison via bias trick
                let bias = i64x2_splat(i64::MIN);
                let sa = v128_xor(a, bias);
                let sb = v128_xor(b, bias);
                i64x2_gt(sa, sb)
            }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{
                <Self as {trait_name}>::simd_gt(b, a)
            }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(a, b))
            }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(b, a))
            }}
        "#});
    }

    // Blend
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}
    "#});

    // Reduce add
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn reduce_add(a: v128) -> {elem} {{
                let arr = <Self as {trait_name}>::to_array(a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}
    "#});

    // Bitwise
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
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
            fn shl_const<const N: i32>(a: v128) -> v128 {{ {shl_fn}(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: v128) -> v128 {{ {unsigned_shr}(a, N as u32) }}
    "#,
        unsigned_shr = if ty.signed {
            format!("{wp}_shr") // unsigned shr for logical
        } else {
            shr_fn.clone()
        },
    });

    if ty.signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {{ {shr_fn}(a, N as u32) }}
        "#});
    }

    // Boolean
    let bitmask_fn = format!("{signed_wp}_bitmask");
    let all_true_fn = format!("{signed_wp}_all_true");

    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(a: v128) -> bool {{ {all_true_fn}(a) }}
            #[inline(always)]
            fn any_true(a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(a: v128) -> u32 {{ {bitmask_fn}(a) as u32 }}
        }}
    "#});

    body
}

fn generate_wasm_polyfill_int_impl(ty: &IntVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let wp = ty.wasm_prefix();
    let sub_count = ty.sub_count();
    let lanes_per_128 = ty.lanes_per_128();

    let signed_wp = format!("{}x{}", ty.signed_elem(), lanes_per_128);

    let binary_op = |fn_name: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{fn_name}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |fn_name: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{fn_name}(a[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", ");
    let z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", ");

    let load_lanes: Vec<String> = (0..sub_count)
        .map(|i| {
            let offset = i * lanes_per_128 * ty.elem_bits / 8;
            format!("v128_load(data.as_ptr().cast::<u8>().add({offset}).cast())")
        })
        .collect();
    let load_body = format!("unsafe {{ [{}] }}", load_lanes.join(", "));

    let store_lines: Vec<String> = (0..sub_count)
        .map(|i| {
            let offset = i * lanes_per_128 * ty.elem_bits / 8;
            format!("v128_store(out.as_mut_ptr().cast::<u8>().add({offset}).cast(), repr[{i}]);")
        })
        .collect();
    let store_body = store_lines.join("\n                    ");

    let blend_items: Vec<String> = (0..sub_count)
        .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
        .collect();
    let blend_body = format!("[{}]", blend_items.join(", "));

    let shl_items: Vec<String> = (0..sub_count)
        .map(|i| format!("{signed_wp}_shl(a[{i}], N as u32)"))
        .collect();
    let shl_body = format!("[{}]", shl_items.join(", "));

    let shr_logical_fn = if ty.signed {
        format!("{wp}_shr") // unsigned shr
    } else {
        format!("{wp}_shr")
    };
    let shr_logical_items: Vec<String> = (0..sub_count)
        .map(|i| format!("{shr_logical_fn}(a[{i}], N as u32)"))
        .collect();
    let shr_logical_body = format!("[{}]", shr_logical_items.join(", "));

    // For u64 polyfill, min/max/comparisons don't exist as WASM intrinsics.
    // Delegate to the native W128 backend trait methods (which have bias-trick polyfills).
    let native_trait = ty.trait_name().replace(
        &format!("x{}", ty.lanes),
        &format!("x{}", ty.lanes_per_128()),
    );

    let trait_binary_op = |method: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                format!("<archmage::Wasm128Token as {native_trait}>::{method}(a[{i}], b[{i}])")
            })
            .collect();
        format!("[{}]", items.join(", "))
    };

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

    let mut code = formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                let v4 = {wp}_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = {wp}_splat(0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                {load_body}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_body}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0{elem}; {lanes}];
                <Self as {trait_name}>::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
    "#,
        add = binary_op(&format!("{signed_wp}_add")),
        sub = binary_op(&format!("{signed_wp}_sub")),
    };

    if ty.has_native_mul() {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
        "#, mul = binary_op(&format!("{signed_wp}_mul"))});
    }

    if ty.signed {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
        "#, neg = unary_op(&format!("{signed_wp}_neg"))});
    }

    // u64 min/max don't exist as WASM intrinsics — delegate to native backend
    if ty.elem_bits == 64 {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
        "#,
            min = trait_binary_op("min"),
            max = trait_binary_op("max"),
        });
    } else {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
        "#,
            min = binary_op(&min_fn),
            max = binary_op(&max_fn),
        });
    }

    if ty.signed && ty.elem_bits <= 16 {
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}
        "#, abs = unary_op(&format!("{signed_wp}_abs"))});
    }

    // u64 unsigned comparisons don't exist as WASM intrinsics — delegate to native backend
    if ty.elem_bits == 64 {
        code.push_str(&formatdoc! {r#"

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
        "#,
            eq = binary_op(&eq_fn),
            ne = binary_op(&ne_fn),
            lt = trait_binary_op("simd_lt"),
            le = trait_binary_op("simd_le"),
            gt = trait_binary_op("simd_gt"),
            ge = trait_binary_op("simd_ge"),
        });
    } else {
        code.push_str(&formatdoc! {r#"

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
        "#,
            eq = binary_op(&eq_fn),
            ne = binary_op(&ne_fn),
            lt = binary_op(&lt_fn),
            le = binary_op(&le_fn),
            gt = binary_op(&gt_fn),
            ge = binary_op(&ge_fn),
        });
    }

    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend_body}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                let arr = <Self as {trait_name}>::to_array(a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
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
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{ {shl_body} }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{ {shr_logical_body} }}
    "#,
        not = unary_op("v128_not"),
        bitand = binary_op("v128_and"),
        bitor = binary_op("v128_or"),
        bitxor = binary_op("v128_xor"),
    });

    if ty.signed {
        let shr_arith_fn = format!("{signed_wp}_shr");
        let shr_arith_items: Vec<String> = (0..sub_count)
            .map(|i| format!("{shr_arith_fn}(a[{i}], N as u32)"))
            .collect();
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_arith}]
            }}
        "#, shr_arith = shr_arith_items.join(", ")});
    }

    // Boolean
    let all_true_fn = format!("{signed_wp}_all_true");
    let bitmask_fn = format!("{signed_wp}_bitmask");

    let all_true_items: Vec<String> = (0..sub_count)
        .map(|i| format!("{all_true_fn}(a[{i}])"))
        .collect();
    let any_true_items: Vec<String> = (0..sub_count)
        .map(|i| format!("v128_any_true(a[{i}])"))
        .collect();

    code.push_str(&formatdoc! {r#"

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
                let mut result = 0u32;
                for i in 0..{sub_count} {{
                    result |= ({bitmask_fn}(a[i]) as u32) << (i * {lanes_per_128});
                }}
                result
            }}
        }}
    "#,
        all_true = all_true_items.join(" && "),
        any_true = any_true_items.join(" || "),
    });

    code
}

// ============================================================================
// Conversion Traits
// ============================================================================

/// Generate additional conversion/bitcast traits for the new types.
pub(super) fn generate_additional_convert_traits() -> String {
    formatdoc! {r#"
        //! Bitcast conversion traits for integer types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Bitcast conversions between i8x16 and u8x16 representations.
        pub trait I8x16Bitcast: super::I8x16Backend + super::U8x16Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i8x16 to u8x16 (reinterpret bits).
            fn bitcast_i8_to_u8(a: <Self as super::I8x16Backend>::Repr) -> <Self as super::U8x16Backend>::Repr;
            /// Bitcast u8x16 to i8x16 (reinterpret bits).
            fn bitcast_u8_to_i8(a: <Self as super::U8x16Backend>::Repr) -> <Self as super::I8x16Backend>::Repr;
        }}

        /// Bitcast conversions between i8x32 and u8x32 representations.
        pub trait I8x32Bitcast: super::I8x32Backend + super::U8x32Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i8x32 to u8x32 (reinterpret bits).
            fn bitcast_i8_to_u8(a: <Self as super::I8x32Backend>::Repr) -> <Self as super::U8x32Backend>::Repr;
            /// Bitcast u8x32 to i8x32 (reinterpret bits).
            fn bitcast_u8_to_i8(a: <Self as super::U8x32Backend>::Repr) -> <Self as super::I8x32Backend>::Repr;
        }}

        /// Bitcast conversions between i16x8 and u16x8 representations.
        pub trait I16x8Bitcast: super::I16x8Backend + super::U16x8Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i16x8 to u16x8 (reinterpret bits).
            fn bitcast_i16_to_u16(a: <Self as super::I16x8Backend>::Repr) -> <Self as super::U16x8Backend>::Repr;
            /// Bitcast u16x8 to i16x8 (reinterpret bits).
            fn bitcast_u16_to_i16(a: <Self as super::U16x8Backend>::Repr) -> <Self as super::I16x8Backend>::Repr;
        }}

        /// Bitcast conversions between i16x16 and u16x16 representations.
        pub trait I16x16Bitcast: super::I16x16Backend + super::U16x16Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i16x16 to u16x16 (reinterpret bits).
            fn bitcast_i16_to_u16(a: <Self as super::I16x16Backend>::Repr) -> <Self as super::U16x16Backend>::Repr;
            /// Bitcast u16x16 to i16x16 (reinterpret bits).
            fn bitcast_u16_to_i16(a: <Self as super::U16x16Backend>::Repr) -> <Self as super::I16x16Backend>::Repr;
        }}

        /// Bitcast conversions between u64x2, i64x2, and f64x2 representations.
        pub trait U64x2Bitcast: super::U64x2Backend + super::I64x2Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u64x2 to i64x2 (reinterpret bits).
            fn bitcast_u64_to_i64(a: <Self as super::U64x2Backend>::Repr) -> <Self as super::I64x2Backend>::Repr;
            /// Bitcast i64x2 to u64x2 (reinterpret bits).
            fn bitcast_i64_to_u64(a: <Self as super::I64x2Backend>::Repr) -> <Self as super::U64x2Backend>::Repr;
        }}

        /// Bitcast conversions between u64x4, i64x4, and f64x4 representations.
        pub trait U64x4Bitcast: super::U64x4Backend + super::I64x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u64x4 to i64x4 (reinterpret bits).
            fn bitcast_u64_to_i64(a: <Self as super::U64x4Backend>::Repr) -> <Self as super::I64x4Backend>::Repr;
            /// Bitcast i64x4 to u64x4 (reinterpret bits).
            fn bitcast_i64_to_u64(a: <Self as super::I64x4Backend>::Repr) -> <Self as super::U64x4Backend>::Repr;
        }}
    "#}
}

/// Generate x86 conversion impls for the new types.
pub(super) fn generate_x86_additional_convert_impls(token: &str) -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "x86_64")]
        impl I8x16Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I8x32Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: __m256i) -> __m256i {{ a }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I16x8Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I16x16Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: __m256i) -> __m256i {{ a }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl U64x2Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: __m128i) -> __m128i {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: __m128i) -> __m128i {{ a }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl U64x4Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: __m256i) -> __m256i {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: __m256i) -> __m256i {{ a }}
        }}
    "#}
}

/// Generate scalar conversion impls for the new types.
pub(super) fn generate_scalar_additional_convert_impls() -> String {
    formatdoc! {r#"

        impl I8x16Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: [i8; 16]) -> [u8; 16] {{
                let mut out = [0u8; 16];
                let mut i = 0;
                while i < 16 {{ out[i] = a[i] as u8; i += 1; }}
                out
            }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: [u8; 16]) -> [i8; 16] {{
                let mut out = [0i8; 16];
                let mut i = 0;
                while i < 16 {{ out[i] = a[i] as i8; i += 1; }}
                out
            }}
        }}

        impl I8x32Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: [i8; 32]) -> [u8; 32] {{
                let mut out = [0u8; 32];
                let mut i = 0;
                while i < 32 {{ out[i] = a[i] as u8; i += 1; }}
                out
            }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: [u8; 32]) -> [i8; 32] {{
                let mut out = [0i8; 32];
                let mut i = 0;
                while i < 32 {{ out[i] = a[i] as i8; i += 1; }}
                out
            }}
        }}

        impl I16x8Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: [i16; 8]) -> [u16; 8] {{
                let mut out = [0u16; 8];
                let mut i = 0;
                while i < 8 {{ out[i] = a[i] as u16; i += 1; }}
                out
            }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: [u16; 8]) -> [i16; 8] {{
                let mut out = [0i16; 8];
                let mut i = 0;
                while i < 8 {{ out[i] = a[i] as i16; i += 1; }}
                out
            }}
        }}

        impl I16x16Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: [i16; 16]) -> [u16; 16] {{
                let mut out = [0u16; 16];
                let mut i = 0;
                while i < 16 {{ out[i] = a[i] as u16; i += 1; }}
                out
            }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: [u16; 16]) -> [i16; 16] {{
                let mut out = [0i16; 16];
                let mut i = 0;
                while i < 16 {{ out[i] = a[i] as i16; i += 1; }}
                out
            }}
        }}

        impl U64x2Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: [u64; 2]) -> [i64; 2] {{
                [a[0] as i64, a[1] as i64]
            }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: [i64; 2]) -> [u64; 2] {{
                [a[0] as u64, a[1] as u64]
            }}
        }}

        impl U64x4Bitcast for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: [u64; 4]) -> [i64; 4] {{
                [a[0] as i64, a[1] as i64, a[2] as i64, a[3] as i64]
            }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: [i64; 4]) -> [u64; 4] {{
                [a[0] as u64, a[1] as u64, a[2] as u64, a[3] as u64]
            }}
        }}
    "#}
}

/// Generate NEON conversion impls for the new types.
pub(super) fn generate_neon_additional_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        impl I8x16Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: int8x16_t) -> uint8x16_t {{
                unsafe {{ vreinterpretq_u8_s8(a) }}
            }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: uint8x16_t) -> int8x16_t {{
                unsafe {{ vreinterpretq_s8_u8(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I8x32Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: [int8x16_t; 2]) -> [uint8x16_t; 2] {{
                unsafe {{ [vreinterpretq_u8_s8(a[0]), vreinterpretq_u8_s8(a[1])] }}
            }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: [uint8x16_t; 2]) -> [int8x16_t; 2] {{
                unsafe {{ [vreinterpretq_s8_u8(a[0]), vreinterpretq_s8_u8(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I16x8Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: int16x8_t) -> uint16x8_t {{
                unsafe {{ vreinterpretq_u16_s16(a) }}
            }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: uint16x8_t) -> int16x8_t {{
                unsafe {{ vreinterpretq_s16_u16(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I16x16Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: [int16x8_t; 2]) -> [uint16x8_t; 2] {{
                unsafe {{ [vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(a[1])] }}
            }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: [uint16x8_t; 2]) -> [int16x8_t; 2] {{
                unsafe {{ [vreinterpretq_s16_u16(a[0]), vreinterpretq_s16_u16(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U64x2Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: uint64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_u64(a) }}
            }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: int64x2_t) -> uint64x2_t {{
                unsafe {{ vreinterpretq_u64_s64(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U64x4Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: [uint64x2_t; 2]) -> [int64x2_t; 2] {{
                unsafe {{ [vreinterpretq_s64_u64(a[0]), vreinterpretq_s64_u64(a[1])] }}
            }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: [int64x2_t; 2]) -> [uint64x2_t; 2] {{
                unsafe {{ [vreinterpretq_u64_s64(a[0]), vreinterpretq_u64_s64(a[1])] }}
            }}
        }}
    "#}
}

/// Generate WASM conversion impls for the new types.
pub(super) fn generate_wasm_additional_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "wasm32")]
        impl I8x16Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I8x32Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i8_to_u8(a: [v128; 2]) -> [v128; 2] {{ a }}
            #[inline(always)]
            fn bitcast_u8_to_i8(a: [v128; 2]) -> [v128; 2] {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I16x8Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I16x16Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i16_to_u16(a: [v128; 2]) -> [v128; 2] {{ a }}
            #[inline(always)]
            fn bitcast_u16_to_i16(a: [v128; 2]) -> [v128; 2] {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U64x2Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: v128) -> v128 {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U64x4Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u64_to_i64(a: [v128; 2]) -> [v128; 2] {{ a }}
            #[inline(always)]
            fn bitcast_i64_to_u64(a: [v128; 2]) -> [v128; 2] {{ a }}
        }}
    "#}
}

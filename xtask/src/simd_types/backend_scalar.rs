//! Unified scalar backend implementation generation (`ScalarToken`).
//!
//! Generates:
//! - `magetypes/src/simd/impls/scalar.rs` (all 20 W128/W256 types + conversions)
//! - `magetypes/src/simd/impls/scalar_w512.rs` (all 10 W512 types + `F32x16Convert`)

use indoc::formatdoc;

use super::types::{ElementType, SimdType};

const CMP_OPS: [(&str, &str); 6] = [
    ("eq", "=="),
    ("ne", "!="),
    ("lt", "<"),
    ("le", "<="),
    ("gt", ">"),
    ("ge", ">="),
];

const BIT_OPS: [(&str, &str); 3] = [("bitand", "&"), ("bitor", "|"), ("bitxor", "^")];

// ============================================================================
// Public File Entry Points
// ============================================================================

/// Generate `magetypes/src/simd/impls/scalar.rs` for all 20 narrow (W128/W256) types.
pub(super) fn generate_scalar_file(narrow: &[SimdType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for ScalarToken (fallback).
        //!
        //! All operations are plain array math. Always available on all platforms.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(clippy::needless_range_loop, clippy::identity_op, clippy::collapsible_else_if, clippy::unnecessary_cast)]

        use crate::simd::backends::*;

    "#};

    // Emit float math wrappers once per float element type to avoid trait method
    // name shadowing inside `impl XxxBackend`.
    for (elem, sqrt_fn, floor_fn, ceil_fn, round_fn) in [
        (
            "f32",
            "crate::scalar_sqrt::sqrtf",
            "crate::scalar::floor_f32",
            "crate::scalar::ceil_f32",
            "crate::scalar::round_f32",
        ),
        (
            "f64",
            "crate::scalar_sqrt::sqrt",
            "crate::scalar::floor_f64",
            "crate::scalar::ceil_f64",
            "crate::scalar::round_f64",
        ),
    ] {
        code.push_str(&formatdoc! {r#"
            // Helpers to avoid trait method name shadowing inside the impl block.
            // Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
            // the trait's associated functions instead of {elem}'s inherent methods.
            #[inline(always)]
            fn {elem}_sqrt(x: {elem}) -> {elem} {{
                {sqrt_fn}(x)
            }}
            #[inline(always)]
            fn {elem}_floor(x: {elem}) -> {elem} {{
                {floor_fn}(x)
            }}
            #[inline(always)]
            fn {elem}_ceil(x: {elem}) -> {elem} {{
                {ceil_fn}(x)
            }}
            #[inline(always)]
            fn {elem}_round(x: {elem}) -> {elem} {{
                {round_fn}(x)
            }}

        "#});
    }

    for &ty in narrow {
        if ty.is_float() {
            code.push_str(&generate_scalar_float_impl(ty));
        } else {
            code.push_str(&generate_scalar_int_impl(ty));
        }
        code.push('\n');
    }

    code.push_str(&generate_scalar_convert_impls());
    code.push_str(&generate_scalar_additional_convert_impls());
    code
}

/// Generate `magetypes/src/simd/impls/scalar_w512.rs` for all 10 W512 types.
pub(super) fn generate_scalar_w512_file(w512: &[SimdType]) -> String {
    let mut code = formatdoc! {r#"
        //! W512 backend implementations for ScalarToken (fallback).
        //!
        //! All operations are plain array math. Gated on `feature = "w512"` at
        //! the module declaration in `impls/mod.rs`.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(
            clippy::needless_range_loop,
            clippy::identity_op,
            clippy::collapsible_else_if,
            clippy::unnecessary_cast
        )]

        use crate::simd::backends::*;

    "#};
    code.push_str(&generate_scalar_f32_convert_impl(16));
    for &ty in w512 {
        if ty.is_float() {
            code.push_str(&generate_scalar_float_impl(ty));
        } else {
            code.push_str(&generate_scalar_int_impl(ty));
        }
        code.push('\n');
    }
    code
}

// ============================================================================
// Float Backend Implementation (W128 / W256 / W512)
// ============================================================================

fn generate_scalar_float_impl(ty: SimdType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let is_f32 = ty.elem == ElementType::F32;
    let is_w512 = ty.is_w512();
    let zero_lit = if is_f32 { "0.0f32" } else { "0.0f64" };
    let abs_mask = if is_f32 {
        "0x7FFF_FFFF"
    } else {
        "0x7FFF_FFFF_FFFF_FFFF"
    };
    let fma_fn = if is_f32 { "fmaf_lanes" } else { "fma_lanes" };
    let (min_ieee, max_ieee) = if is_f32 {
        ("minimumf", "maximumf")
    } else {
        ("minimum", "maximum")
    };
    let sqrt_call = if is_w512 {
        if is_f32 {
            "crate::scalar_sqrt::sqrtf"
        } else {
            "crate::scalar_sqrt::sqrt"
        }
    } else if is_f32 {
        "f32_sqrt"
    } else {
        "f64_sqrt"
    };

    let rsqrt_approx_body = if is_f32 {
        formatdoc! {r#"
            core::array::from_fn(|i| {{
                let x = a[i];
                let seed = f32::from_bits(0x5f37_59df_u32.wrapping_sub(x.to_bits() >> 1));
                let hx = 0.5 * x;
                let y = seed * (1.5 - hx * seed * seed);
                y * (1.5 - hx * y * y)
            }})
        "#}
    } else {
        format!("core::array::from_fn(|i| 1.0 / {sqrt_call}(a[i]))")
    };

    let recip_section = formatdoc! {r#"
        #[inline(always)]
        fn rcp_approx(self, a: {array}) -> {array} {{
            core::array::from_fn(|i| 1.0 / a[i])
        }}
        #[inline(always)]
        fn rsqrt_approx(self, a: {array}) -> {array} {{
            {rsqrt_approx_body}
        }}
        #[inline(always)]
        fn recip(self, a: {array}) -> {array} {{
            core::array::from_fn(|i| 1.0 / a[i])
        }}
        #[inline(always)]
        fn rsqrt(self, a: {array}) -> {array} {{
            core::array::from_fn(|i| 1.0 / {sqrt_call}(a[i]))
        }}
    "#};

    let scalar_internal = if is_f32 {
        super::float_layout::scalar_internal_impl(lanes)
    } else {
        String::new()
    };

    let (splat_body, neg_body, sqrt_body, abs_body, floor_body, ceil_body, round_body, not_body) =
        if is_w512 {
            let (fl, cl, rd) = if is_f32 {
                ("floor_f32", "ceil_f32", "round_f32")
            } else {
                ("floor_f64", "ceil_f64", "round_f64")
            };
            (
                format!("[{zero_lit}; {lanes}].map(|_| v)"),
                "core::array::from_fn(|i| -a[i])".to_string(),
                format!("core::array::from_fn(|i| {sqrt_call}(a[i]))"),
                format!("core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() & {abs_mask}))"),
                format!("a.map(crate::scalar::{fl})"),
                format!("a.map(crate::scalar::{cl})"),
                format!("a.map(crate::scalar::{rd})"),
                format!("core::array::from_fn(|i| {elem}::from_bits(!a[i].to_bits()))"),
            )
        } else {
            (
                format!("[v; {lanes}]"),
                "a.map(|x| -x)".to_string(),
                format!("a.map({elem}_sqrt)"),
                format!("a.map(|x| {elem}::from_bits(x.to_bits() & {abs_mask}))"),
                format!("a.map({elem}_floor)"),
                format!("a.map({elem}_ceil)"),
                format!("a.map({elem}_round)"),
                format!("a.map(|x| {elem}::from_bits(!x.to_bits()))"),
            )
        };

    let mul_sub_portable = if is_w512 {
        let neg_c = (0..lanes)
            .map(|i| format!("-c[{i}]"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("crate::soft_fma::{fma_fn}(a, b, [{neg_c}])")
    } else {
        format!("crate::soft_fma::{fma_fn}(a, b, c.map(|x| -x))")
    };

    let (true_mask, false_mask) = if is_w512 {
        let uint = if is_f32 { "u32" } else { "u64" };
        (format!("{elem}::from_bits(!0{uint})"), zero_lit)
    } else if is_f32 {
        ("f32::from_bits(0xFFFF_FFFF)".to_string(), "0.0")
    } else {
        ("f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)".to_string(), "0.0")
    };

    let blend_body = if is_w512 {
        "core::array::from_fn(|i| if mask[i].to_bits() != 0 { if_true[i] } else { if_false[i] })"
            .to_string()
    } else {
        let sign_shift = if is_f32 { 31 } else { 63 };
        formatdoc! {r#"
            // Check sign bit of mask (all-1s has sign bit set)
            core::array::from_fn(|i| {{
                if (mask[i].to_bits() >> {sign_shift}) != 0 {{
                    if_true[i]
                }} else {{
                    if_false[i]
                }}
            }})
        "#}
    };

    let (reduce_add, reduce_min, reduce_max) = if is_w512 {
        (
            "a.iter().sum()".to_string(),
            format!("a.iter().copied().fold({elem}::INFINITY, {elem}::min)"),
            format!("a.iter().copied().fold({elem}::NEG_INFINITY, {elem}::max)"),
        )
    } else {
        // Adjacent-pair tree for f32 — matches x86 v3 and NEON reduce_add shape (#50).
        let radd = match (ty.elem, lanes) {
            (ElementType::F32, 4) => "(a[0] + a[1]) + (a[2] + a[3])".to_string(),
            (ElementType::F32, 8) => {
                "((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))".to_string()
            }
            _ => (0..lanes)
                .map(|i| format!("a[{i}]"))
                .collect::<Vec<_>>()
                .join(" + "),
        };
        let rmin = "let mut m = a[0];\n        for &v in &a[1..] {\n            m = m.min(v);\n        }\n        m"
            .to_string();
        let rmax = "let mut m = a[0];\n        for &v in &a[1..] {\n            m = m.max(v);\n        }\n        m"
            .to_string();
        (radd, rmin, rmax)
    };

    let (mid_recip, tail_recip) = if is_w512 {
        (format!("\n{recip_section}"), String::new())
    } else {
        (
            String::new(),
            format!("\n    // ====== Approximations ======\n{recip_section}"),
        )
    };

    let (hdr_const, hdr_arith, hdr_math, hdr_cmp, hdr_red, hdr_bit) = if is_w512 {
        ("", "", "", "", "", "")
    } else {
        (
            "\n    // ====== Construction ======\n",
            "\n    // ====== Arithmetic ======\n",
            "\n    // ====== Math ======\n",
            "\n    // ====== Comparisons ======\n",
            "\n    // ====== Reductions ======\n",
            "\n    // ====== Bitwise ======\n",
        )
    };

    let cmps: String = CMP_OPS
        .into_iter()
        .map(|(name, op)| {
            formatdoc! {r#"
                #[inline(always)]
                fn simd_{name}(self, a: {array}, b: {array}) -> {array} {{
                    core::array::from_fn(|i| if a[i] {op} b[i] {{ {true_mask} }} else {{ {false_mask} }})
                }}

            "#}
        })
        .collect();

    let bitops: String = BIT_OPS
        .into_iter()
        .map(|(name, op)| {
            formatdoc! {r#"

                #[inline(always)]
                fn {name}(self, a: {array}, b: {array}) -> {array} {{
                    core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() {op} b[i].to_bits()))
                }}
            "#}
        })
        .collect();

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};
        {hdr_const}
            #[inline(always)]
            fn splat(self, v: {elem}) -> {array} {{
                {splat_body}
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [{zero_lit}; {lanes}]
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
        {hdr_arith}
            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] + b[i])
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] - b[i])
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i])
            }}

            #[inline(always)]
            fn div(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] / b[i])
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                {neg_body}
            }}
        {hdr_math}
            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].min(b[i]))
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].max(b[i]))
            }}

            #[inline(always)]
            fn minimum(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| crate::scalar_extrema::{min_ieee}(a[i], b[i]))
            }}

            #[inline(always)]
            fn maximum(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| crate::scalar_extrema::{max_ieee}(a[i], b[i]))
            }}

            #[inline(always)]
            fn min_fast(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn max_fast(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn sqrt(self, a: {array}) -> {array} {{
                {sqrt_body}
            }}
        {mid_recip}
            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                {abs_body}
            }}

            #[inline(always)]
            fn floor(self, a: {array}) -> {array} {{
                {floor_body}
            }}

            #[inline(always)]
            fn ceil(self, a: {array}) -> {array} {{
                {ceil_body}
            }}

            #[inline(always)]
            fn round(self, a: {array}) -> {array} {{
                {round_body}
            }}

            #[inline(always)]
            fn mul_add(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i] + c[i])
            }}

            #[inline(always)]
            fn mul_sub(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i] - c[i])
            }}

            #[inline(always)]
            fn mul_add_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                crate::soft_fma::{fma_fn}(a, b, c)
            }}

            #[inline(always)]
            fn mul_sub_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_sub_portable}
            }}
        {hdr_cmp}
        {cmps}
            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                {blend_body}
            }}
        {hdr_red}
            #[inline(always)]
            fn reduce_add(self, a: {array}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {array}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(self, a: {array}) -> {elem} {{
                {reduce_max}
            }}
        {tail_recip}
        {hdr_bit}
            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                {not_body}
            }}
        {bitops}
        }}{scalar_internal}
    "#}
}

// ============================================================================
// Integer Backend Implementation (W128 / W256 / W512)
// ============================================================================

/// `a[i]` reinterpreted as `via` for a wrapping operation, then back as `elem`.
/// Emits no casts when the two types match (`trivial_numeric_casts`).
fn via(elem: &str, via: &str, op: &str) -> String {
    if elem == via {
        format!("a[i].{op}")
    } else {
        format!("(a[i] as {via}).{op} as {elem}")
    }
}

fn generate_scalar_int_impl(ty: SimdType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let elem_bits = ty.elem_bits();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let signed_elem = ty.elem.signed_int_name();
    let unsigned_elem = ty.elem.unsigned_int_name();
    let is_w512 = ty.is_w512();
    let is_signed = ty.is_signed();
    let is_narrow_32_64 = !is_w512
        && matches!(
            ty.elem,
            ElementType::I32 | ElementType::U32 | ElementType::I64
        );
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "ScalarToken");

    let zero_lit = if is_w512 {
        "0".to_string()
    } else {
        format!("0{elem}")
    };

    let mask_true = if is_w512 {
        format!("!0{elem}")
    } else {
        match ty.elem {
            ElementType::I8 | ElementType::I16 | ElementType::I32 | ElementType::I64 => {
                "-1".to_string()
            }
            ElementType::U8 => "0xFF".to_string(),
            ElementType::U16 => "0xFFFF".to_string(),
            ElementType::U32 => "u32::MAX".to_string(),
            ElementType::U64 => "u64::MAX".to_string(),
            _ => unreachable!(),
        }
    };

    let mul_section = if matches!(elem_bits, 16 | 32) {
        formatdoc! {r#"

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
            }}
        "#}
    } else {
        String::new()
    };

    let neg_section = if is_signed || is_w512 {
        let body = if is_narrow_32_64 {
            format!("a.map({elem}::wrapping_neg)")
        } else if is_signed {
            "core::array::from_fn(|i| a[i].wrapping_neg())".to_string()
        } else {
            format!("core::array::from_fn(|i| (0{elem}).wrapping_sub(a[i]))")
        };
        formatdoc! {r#"

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                {body}
            }}
        "#}
    } else {
        String::new()
    };

    let abs_section = if is_signed {
        let body = if is_narrow_32_64 {
            format!("a.map({elem}::wrapping_abs)")
        } else {
            "core::array::from_fn(|i| a[i].wrapping_abs())".to_string()
        };
        formatdoc! {r#"

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                {body}
            }}
        "#}
    } else {
        String::new()
    };

    let reduce_add_body = if is_narrow_32_64 {
        (0..lanes)
            .map(|i| format!("a[{i}]"))
            .collect::<Vec<_>>()
            .join(".wrapping_add(")
            + &")".repeat(lanes - 1)
    } else {
        format!("a.iter().copied().fold(0{elem}, {elem}::wrapping_add)")
    };

    let not_body = if is_narrow_32_64 {
        "a.map(|x| !x)"
    } else {
        "core::array::from_fn(|i| !a[i])"
    };

    let shifts_section = if is_w512 {
        let shl = format!(
            "core::array::from_fn(|i| if N < {elem_bits} {{ {} }} else {{ 0 }})",
            via(elem, unsigned_elem, "wrapping_shl(N as u32)")
        );
        let shr_arith = if is_signed {
            format!(
                "core::array::from_fn(|i| if N < {elem_bits} {{ a[i].wrapping_shr(N as u32) }} else if a[i] < 0 {{ -1 }} else {{ 0 }})"
            )
        } else {
            format!(
                "core::array::from_fn(|i| if N < {elem_bits} {{ a[i].wrapping_shr(N as u32) }} else {{ 0 }})"
            )
        };
        let shr_logical = format!(
            "core::array::from_fn(|i| if N < {elem_bits} {{ {} }} else {{ 0 }})",
            via(elem, unsigned_elem, "wrapping_shr(N as u32)")
        );
        formatdoc! {r#"

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_arith}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_logical}
            }}
        "#}
    } else if is_narrow_32_64 {
        if is_signed {
            formatdoc! {r#"

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
                    a.map(|x| ((x as {unsigned_elem}) >> N) as {elem})
                }}
            "#}
        } else {
            formatdoc! {r#"

                #[inline(always)]
                fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                    a.map(|x| x << N)
                }}

                #[inline(always)]
                fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                    a.map(|x| x >> N)
                }}
            "#}
        }
    } else {
        let shr_logical = if is_signed {
            format!(
                "core::array::from_fn(|i| (a[i] as {unsigned_elem}).wrapping_shr(N as u32) as {elem})"
            )
        } else {
            "core::array::from_fn(|i| a[i].wrapping_shr(N as u32))".to_string()
        };
        let mut s = formatdoc! {r#"

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_shl(N as u32))
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_logical}
            }}
        "#};
        if is_signed {
            s.push_str(&formatdoc! {r#"

                #[inline(always)]
                fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{
                    core::array::from_fn(|i| a[i].wrapping_shr(N as u32))
                }}
            "#});
        }
        s
    };

    let uniform_shifts_section = if matches!(elem_bits, 16 | 32) {
        let max_sh = elem_bits - 1;
        if is_w512 {
            let wrapping_shl_via = via(elem, unsigned_elem, "wrapping_shl(count)");
            let wrapping_shr_via = via(elem, unsigned_elem, "wrapping_shr(count)");
            let shr_arith = if is_signed {
                format!(
                    "core::array::from_fn(|i| a[i].wrapping_shr(if count > {max_sh} {{ {max_sh} }} else {{ count }}))"
                )
            } else {
                format!(
                    "core::array::from_fn(|i| if count >= {elem_bits} {{ 0 }} else {{ a[i].wrapping_shr(count) }})"
                )
            };
            formatdoc! {r#"

                #[inline(always)]
                fn shl_uniform(self, a: {array}, count: u32) -> {array} {{
                    core::array::from_fn(|i| if count >= {elem_bits} {{ 0 }} else {{ {wrapping_shl_via} }})
                }}

                #[inline(always)]
                fn shr_logical_uniform(self, a: {array}, count: u32) -> {array} {{
                    core::array::from_fn(|i| if count >= {elem_bits} {{ 0 }} else {{ {wrapping_shr_via} }})
                }}

                #[inline(always)]
                fn shr_arithmetic_uniform(self, a: {array}, count: u32) -> {array} {{
                    {shr_arith}
                }}
            "#}
        } else {
            let hdr = if elem_bits == 32 {
                "\n    // ====== Uniform variable shifts ======\n"
            } else {
                ""
            };
            let shr_logical_lane = if is_signed {
                format!("((x as {unsigned_elem}).wrapping_shr(count)) as {elem}")
            } else {
                "x.wrapping_shr(count)".to_string()
            };
            let mut s = formatdoc! {r#"
                {hdr}
                #[inline(always)]
                fn shl_uniform(self, a: {array}, count: u32) -> {array} {{
                    a.map(|x| if count >= {elem_bits} {{ 0 }} else {{ x.wrapping_shl(count) }})
                }}

                #[inline(always)]
                fn shr_logical_uniform(self, a: {array}, count: u32) -> {array} {{
                    a.map(|x| if count >= {elem_bits} {{ 0 }} else {{ {shr_logical_lane} }})
                }}
            "#};
            if is_signed {
                let arith_body = if elem_bits == 32 {
                    "let c = if count > 31 { 31 } else { count };\n        a.map(|x| x.wrapping_shr(c))"
                        .to_string()
                } else {
                    format!(
                        "// Clamping to lane_bits - 1 *is* the sign fill.\n        a.map(|x| x.wrapping_shr(if count > {max_sh} {{ {max_sh} }} else {{ count }}))"
                    )
                };
                s.push_str(&formatdoc! {r#"

                    #[inline(always)]
                    fn shr_arithmetic_uniform(self, a: {array}, count: u32) -> {array} {{
                        {arith_body}
                    }}
                "#});
            }
            s
        }
    } else {
        String::new()
    };

    let saturating_section = if matches!(elem_bits, 8 | 16) {
        formatdoc! {r#"

            #[inline(always)]
            fn saturating_add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].saturating_add(b[i]))
            }}

            #[inline(always)]
            fn saturating_sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].saturating_sub(b[i]))
            }}
        "#}
    } else {
        String::new()
    };

    let sign_shift = elem_bits - 1;
    let bitmask_ret = if is_w512 { "u64" } else { "u32" };
    let (all_true_body, any_true_body, bitmask_body) = if is_w512 {
        let neg_v = if is_signed {
            "v < 0".to_string()
        } else {
            format!("(v as {signed_elem}) < 0")
        };
        let neg_ai = if is_signed {
            "a[i] < 0".to_string()
        } else {
            format!("(a[i] as {signed_elem}) < 0")
        };
        (
            format!("a.iter().all(|&v| {neg_v})"),
            format!("a.iter().any(|&v| {neg_v})"),
            format!(
                "let mut mask = 0u64;\n        for i in 0..{lanes} {{\n            if {neg_ai} {{\n                mask |= 1u64 << i;\n            }}\n        }}\n        mask"
            ),
        )
    } else if is_narrow_32_64 {
        let all_t = (0..lanes)
            .map(|i| format!("(a[{i}] >> {sign_shift}) != 0"))
            .collect::<Vec<_>>()
            .join(" && ");
        let any_t = (0..lanes)
            .map(|i| format!("(a[{i}] >> {sign_shift}) != 0"))
            .collect::<Vec<_>>()
            .join(" || ");
        let bm = (0..lanes)
            .map(|i| match ty.elem {
                ElementType::I32 => {
                    if i == 0 {
                        "((a[0] as u32) >> 31)".to_string()
                    } else {
                        format!("(((a[{i}] as u32) >> 31) << {i})")
                    }
                }
                ElementType::U32 => {
                    if i == 0 {
                        "(a[0] >> 31)".to_string()
                    } else {
                        format!("((a[{i}] >> 31) << {i})")
                    }
                }
                ElementType::I64 => {
                    if i == 0 {
                        "((a[0] as u64) >> 63) as u32".to_string()
                    } else {
                        format!("((((a[{i}] as u64) >> 63) as u32) << {i})")
                    }
                }
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
            .join(" | ");
        (all_t, any_t, bm)
    } else {
        (
            format!("a.iter().all(|&v| (v >> {sign_shift}) != 0)"),
            format!("a.iter().any(|&v| (v >> {sign_shift}) != 0)"),
            formatdoc! {r#"
                let mut mask = 0u32;
                for (i, &v) in a.iter().enumerate() {{
                    mask |= ((v >> {sign_shift}) as u32 & 1) << i;
                }}
                mask
            "#},
        )
    };

    let (hdr_const, hdr_arith, hdr_math, hdr_cmp, hdr_red, hdr_bit, hdr_sh, hdr_bool) =
        if is_narrow_32_64 {
            (
                "\n    // ====== Construction ======\n",
                "\n    // ====== Arithmetic ======\n",
                "\n    // ====== Math ======\n",
                "\n    // ====== Comparisons ======\n",
                "\n    // ====== Reductions ======\n",
                "\n    // ====== Bitwise ======\n",
                "\n    // ====== Shifts ======\n",
                "\n    // ====== Boolean ======\n",
            )
        } else {
            ("", "", "", "", "", "", "", "")
        };

    let cmps: String = CMP_OPS
        .into_iter()
        .map(|(name, op)| {
            formatdoc! {r#"
                #[inline(always)]
                fn simd_{name}(self, a: {array}, b: {array}) -> {array} {{
                    core::array::from_fn(|i| if a[i] {op} b[i] {{ {mask_true} }} else {{ 0 }})
                }}

            "#}
        })
        .collect();

    let bitops: String = BIT_OPS
        .into_iter()
        .map(|(name, op)| {
            formatdoc! {r#"

                #[inline(always)]
                fn {name}(self, a: {array}, b: {array}) -> {array} {{
                    core::array::from_fn(|i| a[i] {op} b[i])
                }}
            "#}
        })
        .collect();

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};
        {hdr_const}
            #[inline(always)]
            fn splat(self, v: {elem}) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [{zero_lit}; {lanes}]
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
        {hdr_arith}
            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_add(b[i]))
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
            }}
        {mul_section}{neg_section}{hdr_math}
            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
            }}
        {abs_section}{hdr_cmp}
        {cmps}
            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}
        {hdr_red}
            #[inline(always)]
            fn reduce_add(self, a: {array}) -> {elem} {{
                {reduce_add_body}
            }}
        {hdr_bit}
            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                {not_body}
            }}
        {bitops}
        {hdr_sh}{shifts_section}{uniform_shifts_section}{saturating_section}{hdr_bool}
            #[inline(always)]
            fn all_true(self, a: {array}) -> bool {{
                {all_true_body}
            }}

            #[inline(always)]
            fn any_true(self, a: {array}) -> bool {{
                {any_true_body}
            }}

            #[inline(always)]
            fn bitmask(self, a: {array}) -> {bitmask_ret} {{
                {bitmask_body}
            }}
        {integer_methods}
        }}
    "#}
}

// ============================================================================
// Scalar Conversion & Bitcast Implementations
// ============================================================================

fn generate_scalar_f32_convert_impl(lanes: usize) -> String {
    let f_array = format!("[f32; {lanes}]");
    let i_array = format!("[i32; {lanes}]");
    let trait_name = format!("F32x{lanes}Convert");

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: {f_array}) -> {i_array} {{
                a.map(|x| x.to_bits() as i32)
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                a.map(|x| f32::from_bits(x as u32))
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: {f_array}) -> {i_array} {{
                a.map(|x| x as i32)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: {f_array}) -> {i_array} {{
                a.map(|x| crate::scalar::round_f32(x) as i32)
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                a.map(|x| x as f32)
            }}
        }}

    "#}
}

fn generate_scalar_convert_impls() -> String {
    let mut code = String::new();

    for lanes in [4, 8] {
        code.push_str(&generate_scalar_f32_convert_impl(lanes));
    }

    for lanes in [4, 8] {
        code.push_str(&formatdoc! {r#"
            impl U32x{lanes}Bitcast for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_u32_to_i32(self, a: [u32; {lanes}]) -> [i32; {lanes}] {{
                    a.map(|x| x as i32)
                }}

                #[inline(always)]
                fn bitcast_i32_to_u32(self, a: [i32; {lanes}]) -> [u32; {lanes}] {{
                    a.map(|x| x as u32)
                }}
            }}

        "#});
    }

    for lanes in [2, 4] {
        code.push_str(&formatdoc! {r#"
            impl I64x{lanes}Bitcast for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_i64_to_f64(self, a: [i64; {lanes}]) -> [f64; {lanes}] {{
                    a.map(|x| f64::from_bits(x as u64))
                }}

                #[inline(always)]
                fn bitcast_f64_to_i64(self, a: [f64; {lanes}]) -> [i64; {lanes}] {{
                    a.map(|x| x.to_bits() as i64)
                }}
            }}

        "#});
    }

    code
}

fn generate_scalar_additional_convert_impls() -> String {
    let mut code = String::new();

    for (trait_prefix, a, b, lanes) in [
        ("I8x16", "i8", "u8", 16usize),
        ("I8x32", "i8", "u8", 32),
        ("I16x8", "i16", "u16", 8),
        ("I16x16", "i16", "u16", 16),
    ] {
        code.push_str(&formatdoc! {r#"

            impl {trait_prefix}Bitcast for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_{a}_to_{b}(self, a: [{a}; {lanes}]) -> [{b}; {lanes}] {{
                    let mut out = [0{b}; {lanes}];
                    let mut i = 0;
                    while i < {lanes} {{
                        out[i] = a[i] as {b};
                        i += 1;
                    }}
                    out
                }}
                #[inline(always)]
                fn bitcast_{b}_to_{a}(self, a: [{b}; {lanes}]) -> [{a}; {lanes}] {{
                    let mut out = [0{a}; {lanes}];
                    let mut i = 0;
                    while i < {lanes} {{
                        out[i] = a[i] as {a};
                        i += 1;
                    }}
                    out
                }}
            }}
        "#});
    }

    for lanes in [2usize, 4] {
        let to_i64 = (0..lanes)
            .map(|i| format!("a[{i}] as i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let to_u64 = (0..lanes)
            .map(|i| format!("a[{i}] as u64"))
            .collect::<Vec<_>>()
            .join(", ");
        code.push_str(&formatdoc! {r#"

            impl U64x{lanes}Bitcast for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_u64_to_i64(self, a: [u64; {lanes}]) -> [i64; {lanes}] {{
                    [{to_i64}]
                }}
                #[inline(always)]
                fn bitcast_i64_to_u64(self, a: [i64; {lanes}]) -> [u64; {lanes}] {{
                    [{to_u64}]
                }}
            }}
        "#});
    }

    code
}

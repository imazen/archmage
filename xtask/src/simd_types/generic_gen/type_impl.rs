//! Generator for `*_impl.rs` files — one per SIMD type.
//!
//! Each file defines the struct, method impls, operator impls, scalar broadcasts,
//! index/debug/from-array impls, cross-type conversions, and platform-specific impls.

use indoc::formatdoc;

use super::*;
use crate::simd_types::types::{ElementType, SimdType, SimdWidth};

/// Generate a complete `*_impl.rs` file for one SIMD type.
pub(super) fn gen_type_impl(ty: &SimdType) -> String {
    let mut code = String::new();
    code.push_str(&gen_header(ty));
    code.push_str(&gen_struct(ty));
    code.push_str(&gen_methods(ty));
    code.push_str(&gen_operators(ty));
    code.push_str(&gen_assign_operators(ty));
    code.push_str(&gen_scalar_broadcast(ty));
    code.push_str(&gen_index(ty));
    code.push_str(&gen_from_array(ty));
    code.push_str(&gen_debug(ty));
    code.push_str(&gen_cross_type(ty));
    code.push_str(&gen_platform(ty));
    code.push_str(&gen_popcnt(ty));
    code
}

// ============================================================================
// Header (module doc + imports)
// ============================================================================

fn gen_header(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    let token_examples = match ty.width {
        SimdWidth::W512 => format!("`X64V4Token`, `ScalarToken`"),
        _ => format!("`X64V3Token`, `NeonToken`, `ScalarToken`"),
    };

    let example_block = if ty.elem.is_float() && lanes >= 8 {
        let backend_module = match ty.width {
            SimdWidth::W512 => "x64v4",
            _ => "x64v3",
        };
        formatdoc! {"
            //!
            //! # Example
            //!
            //! ```ignore
            //! use magetypes::simd::backends::{{{backend}, {backend_module}}};
            //! use magetypes::simd::generic::{name};
            //!
            //! fn sum<T: {backend}>(token: T, data: &[{elem}]) -> {elem} {{
            //!     let mut acc = {name}::<T>::zero(token);
            //!     for chunk in data.chunks_exact({lanes}) {{
            //!         acc = acc + {name}::<T>::load(token, chunk.try_into().unwrap());
            //!     }}
            //!     acc.reduce_add()
            //! }}
            //! ```
        "}
    } else {
        String::new()
    };

    let mut ops = vec![
        "Add",
        "AddAssign",
        "BitAnd",
        "BitAndAssign",
        "BitOr",
        "BitOrAssign",
        "BitXor",
        "BitXorAssign",
    ];
    if has_div(ty.elem) {
        ops.push("Div");
        ops.push("DivAssign");
    }
    ops.push("Index");
    ops.push("IndexMut");
    if has_mul(ty.elem) {
        ops.push("Mul");
        ops.push("MulAssign");
    }
    if has_neg(ty.elem) {
        ops.push("Neg");
    }
    ops.push("Sub");
    ops.push("SubAssign");
    ops.sort();

    let ops_str = format_ops_import(&ops);

    formatdoc! {"
        //! Generic `{name}<T>` — {lanes}-lane {elem} SIMD vector parameterized by backend.
        //!
        //! `T` is a token type (e.g., {token_examples})
        //! that determines the platform-native representation and intrinsics used.
        //! The struct delegates all operations to the [`{backend}`] trait.
        {example_block}
        #![allow(clippy::should_implement_trait)]

        {ops_str}
        use crate::simd::backends::{backend};

    "}
}

/// Format the ops import to match the handwritten pattern.
fn format_ops_import(ops: &[&str]) -> String {
    let items: Vec<String> = ops.iter().map(|s| s.to_string()).collect();

    let single_line = format!("use core::ops::{{{}}};", items.join(", "));
    if single_line.len() <= 100 {
        return single_line + "\n";
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();

    for (i, item) in items.iter().enumerate() {
        let separator = if i < items.len() - 1 { ", " } else { "," };
        let addition = format!("{item}{separator}");

        if current_line.is_empty() {
            current_line = addition;
        } else if format!("    {current_line}{addition}").len() <= 100 {
            current_line.push_str(&addition);
        } else {
            lines.push(current_line);
            current_line = addition;
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    let mut result = "use core::ops::{\n".to_string();
    for line in &lines {
        result.push_str(&format!("    {line}\n"));
    }
    result.push_str("};\n");
    result
}

// ============================================================================
// Struct definition
// ============================================================================

fn gen_struct(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    let repr_doc = match ty.width {
        SimdWidth::W128 => {
            let x86_repr = x86_repr_hint(ty);
            let arm_repr = arm_repr_hint(ty);
            format!("`{x86_repr}` on x86, `{arm_repr}` on ARM")
        }
        SimdWidth::W256 => {
            let x86_repr = x86_repr_hint(ty);
            format!("`{x86_repr}` on AVX2, `[{elem}; {lanes}]` on scalar")
        }
        SimdWidth::W512 => {
            let x86_repr = x86_repr_hint(ty);
            format!("`{x86_repr}` on AVX-512, `[{elem}; {lanes}]` on scalar")
        }
    };

    let note_section = if matches!(ty.elem, ElementType::I64) {
        formatdoc! {"
            ///
            /// # Note
            ///
            /// 64-bit integer SIMD has limited native support: no hardware multiply on
            /// AVX2/NEON/WASM, and arithmetic right shift requires AVX-512 on x86.
            /// Operations like `min`, `max`, and `abs` are polyfilled where needed.
        "}
    } else if matches!(ty.elem, ElementType::U64) {
        formatdoc! {"
            ///
            /// # Note
            ///
            /// 64-bit integer SIMD has limited native support: no hardware multiply on
            /// AVX2/NEON/WASM.
        "}
    } else {
        String::new()
    };

    let phantom_comment = if ty.elem.is_float() && lanes >= 8 {
        format!("\n// PhantomData is ZST, so {name}<T> has the same size as T::Repr.\n")
    } else {
        String::new()
    };

    formatdoc! {"
        /// {lanes}-lane {elem} SIMD vector, generic over backend `T`.
        ///
        /// `T` is a token type that proves CPU support for the required SIMD features.
        /// The inner representation is `T::Repr` (e.g., {repr_doc}).
        ///
        /// **The token is stored** (as a zero-sized field) so methods receiving
        /// `self: {name}<T>` can re-supply it to backend operations that
        /// require a token value (e.g. `T::splat(token, v)`). This carries the
        /// token-as-feature-proof guarantee through every method call without
        /// runtime overhead — `T` is ZST, so `sizeof({name}<T>) == sizeof(T::Repr)`
        /// and `align_of({name}<T>) == align_of(T::Repr)` under `#[repr(C)]`.
        ///
        /// # Layout
        ///
        /// `#[repr(C)]` with a ZST trailing field: `T::Repr` lives at offset 0
        /// and `T` is a 0-byte tail. Bitcasts between `{name}<T>` values of
        /// different element-types are sound when the Repr types share a layout
        /// (e.g. `__m128` and `__m128i` are both 16-byte aligned 128-bit values).
        /// `#[repr(transparent)]` cannot be used because Rust cannot prove at
        /// the struct definition site that a generic `T` is a 1-ZST.
        ///
        /// Construction requires a token value to prove CPU support at runtime.
        {note_section}#[derive(Clone, Copy)]
        #[repr(C)]
        pub struct {name}<T: {backend}>(pub(crate) T::Repr, pub(crate) T);
        {phantom_comment}
    "}
}

// ============================================================================
// Methods (main impl block)
// ============================================================================

fn gen_methods(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);
    let bitmask_ty = bitmask_type(ty.width);
    let elem_bits = ty.elem.size_bytes() * 8;
    let high_bit = high_bit_doc(ty.elem);

    let mut code = format!("impl<T: {backend}> {name}<T> {{\n");

    code.push_str(&formatdoc! {"
        \x20   /// Number of {elem} lanes.
            pub const LANES: usize = {lanes};

    "});

    // ====== Construction ======
    code.push_str("    // ====== Construction (token-gated) ======\n\n");
    code.push_str(&gen_construction(elem, lanes));
    code.push_str(&gen_partition_slice(ty));
    code.push_str(&gen_partition_slice_mut(ty));

    // ====== Accessors ======
    code.push_str("    // ====== Accessors ======\n\n");
    code.push_str(&gen_accessors(elem, lanes));

    // ====== Math ======
    code.push_str("    // ====== Math ======\n\n");
    code.push_str(&gen_math(ty));

    // ====== Comparisons ======
    code.push_str("    // ====== Comparisons ======\n\n");
    let signedness = signedness_doc(ty.elem);
    code.push_str(&gen_comparisons(signedness));

    // ====== Reductions ======
    code.push_str("    // ====== Reductions ======\n\n");
    let wrapping_suffix = if ty.elem.is_float() {
        ""
    } else {
        " (wrapping)"
    };
    code.push_str(&formatdoc! {"
        \x20   /// Sum all {lanes} lanes{wrapping_suffix}.
            #[inline(always)]
            pub fn reduce_add(self) -> {elem} {{
                T::reduce_add(self.0)
            }}

    "});
    if has_reduce_min_max(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Minimum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_min(self) -> {elem} {{
                    T::reduce_min(self.0)
                }}

                /// Maximum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_max(self) -> {elem} {{
                    T::reduce_max(self.0)
                }}

        "});
    }

    // ====== Approximations ======
    if has_approx(ty.elem) {
        code.push_str("    // ====== Approximations ======\n\n");
        code.push_str(&gen_approximations());
    }

    // ====== Shifts ======
    if has_shifts(ty.elem) {
        code.push_str("    // ====== Shifts ======\n\n");
        code.push_str(&gen_shifts(ty));
    }

    // ====== Bitwise ======
    code.push_str("    // ====== Bitwise ======\n\n");
    code.push_str(&formatdoc! {"
        \x20   /// Bitwise NOT.
            #[inline(always)]
            pub fn not(self) -> Self {{
                Self(T::not(self.0), self.1)
            }}

    "});

    // ====== Boolean ======
    if has_boolean(ty.elem) {
        code.push_str("    // ====== Boolean ======\n\n");
        code.push_str(&formatdoc! {"
            \x20   /// True if all lanes have their {high_bit} set (all-1s mask).
                #[inline(always)]
                pub fn all_true(self) -> bool {{
                    T::all_true(self.0)
                }}

                /// True if any lane has its {high_bit} set.
                #[inline(always)]
                pub fn any_true(self) -> bool {{
                    T::any_true(self.0)
                }}

                /// Extract the high bit of each {elem_bits}-bit lane as a bitmask.
                #[inline(always)]
                pub fn bitmask(self) -> {bitmask_ty} {{
                    T::bitmask(self.0)
                }}

        "});
    }

    code.push_str("}\n\n");
    code
}

fn gen_construction(elem: &str, lanes: usize) -> String {
    // Token threading: backend trait construction methods now take `self`,
    // and the generic struct stores `T` (ZST), so we pass the user's token
    // through to the backend AND store it for later operator/method use.
    formatdoc! {"
        \x20   /// Broadcast scalar to all {lanes} lanes.
            #[inline(always)]
            pub fn splat(token: T, v: {elem}) -> Self {{
                Self(T::splat(token, v), token)
            }}

            /// All lanes zero.
            #[inline(always)]
            pub fn zero(token: T) -> Self {{
                Self(T::zero(token), token)
            }}

            /// Load from a `[{elem}; {lanes}]` array.
            #[inline(always)]
            pub fn load(token: T, data: &[{elem}; {lanes}]) -> Self {{
                Self(T::load(token, data), token)
            }}

            /// Create from array (zero-cost where possible).
            #[inline(always)]
            pub fn from_array(token: T, arr: [{elem}; {lanes}]) -> Self {{
                Self(T::from_array(token, arr), token)
            }}

            /// Create from slice. Panics if `slice.len() < {lanes}`.
            #[inline(always)]
            pub fn from_slice(token: T, slice: &[{elem}]) -> Self {{
                let arr: [{elem}; {lanes}] = slice[..{lanes}].try_into().unwrap();
                Self(T::from_array(token, arr), token)
            }}

    "}
}

fn gen_accessors(elem: &str, lanes: usize) -> String {
    formatdoc! {"
        \x20   /// Store to array.
            #[inline(always)]
            pub fn store(self, out: &mut [{elem}; {lanes}]) {{
                T::store(self.0, out);
            }}

            /// Convert to array.
            #[inline(always)]
            pub fn to_array(self) -> [{elem}; {lanes}] {{
                T::to_array(self.0)
            }}

            /// Get the underlying platform representation.
            #[inline(always)]
            pub fn into_repr(self) -> T::Repr {{
                self.0
            }}

            /// Wrap a platform representation (token-gated).
            #[inline(always)]
            pub fn from_repr(token: T, repr: T::Repr) -> Self {{
                Self(repr, token)
            }}

            /// Wrap a repr with a token. Used by cross-type/cross-width helpers
            /// in `simd::generic::*` where the token is already proven by the
            /// caller's wider input type.
            #[inline(always)]
            #[allow(dead_code)]
            pub(crate) fn from_repr_unchecked(token: T, repr: T::Repr) -> Self {{
                Self(repr, token)
            }}

    "}
}

fn gen_math(ty: &SimdType) -> String {
    if ty.elem.is_float() {
        gen_float_math()
    } else {
        gen_int_math(ty)
    }
}

fn gen_float_math() -> String {
    formatdoc! {"
        \x20   /// Lane-wise minimum.
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
                Self(T::min(self.0, other.0), self.1)
            }}

            /// Lane-wise maximum.
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
                Self(T::max(self.0, other.0), self.1)
            }}

            /// Clamp between lo and hi.
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                Self(T::clamp(self.0, lo.0, hi.0), self.1)
            }}

            /// Square root.
            #[inline(always)]
            pub fn sqrt(self) -> Self {{
                Self(T::sqrt(self.0), self.1)
            }}

            /// Absolute value.
            #[inline(always)]
            pub fn abs(self) -> Self {{
                Self(T::abs(self.0), self.1)
            }}

            /// Round toward negative infinity.
            #[inline(always)]
            pub fn floor(self) -> Self {{
                Self(T::floor(self.0), self.1)
            }}

            /// Round toward positive infinity.
            #[inline(always)]
            pub fn ceil(self) -> Self {{
                Self(T::ceil(self.0), self.1)
            }}

            /// Round to nearest integer.
            #[inline(always)]
            pub fn round(self) -> Self {{
                Self(T::round(self.0), self.1)
            }}

            /// Fused multiply-add: `self * a + b`.
            #[inline(always)]
            pub fn mul_add(self, a: Self, b: Self) -> Self {{
                Self(T::mul_add(self.0, a.0, b.0), self.1)
            }}

            /// Fused multiply-sub: `self * a - b`.
            #[inline(always)]
            pub fn mul_sub(self, a: Self, b: Self) -> Self {{
                Self(T::mul_sub(self.0, a.0, b.0), self.1)
            }}

    "}
}

fn gen_int_math(ty: &SimdType) -> String {
    let unsigned_suffix = if !ty.elem.is_signed() {
        " (unsigned)"
    } else {
        ""
    };

    let mut code = formatdoc! {"
        \x20   /// Lane-wise minimum{unsigned_suffix}.
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
                Self(T::min(self.0, other.0), self.1)
            }}

            /// Lane-wise maximum{unsigned_suffix}.
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
                Self(T::max(self.0, other.0), self.1)
            }}

    "};

    if has_abs(ty.elem) && !ty.elem.is_float() {
        code.push_str(&formatdoc! {"
            \x20   /// Lane-wise absolute value.
                #[inline(always)]
                pub fn abs(self) -> Self {{
                    Self(T::abs(self.0), self.1)
                }}

        "});
    }

    code.push_str(&formatdoc! {"
        \x20   /// Clamp between lo and hi.
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                Self(T::clamp(self.0, lo.0, hi.0), self.1)
            }}

    "});

    code
}

fn gen_comparisons(signedness: &str) -> String {
    formatdoc! {"
        \x20   /// Lane-wise equality (returns mask).
            #[inline(always)]
            pub fn simd_eq(self, other: Self) -> Self {{
                Self(T::simd_eq(self.0, other.0), self.1)
            }}

            /// Lane-wise inequality (returns mask).
            #[inline(always)]
            pub fn simd_ne(self, other: Self) -> Self {{
                Self(T::simd_ne(self.0, other.0), self.1)
            }}

            /// Lane-wise less-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> Self {{
                Self(T::simd_lt(self.0, other.0), self.1)
            }}

            /// Lane-wise less-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_le(self, other: Self) -> Self {{
                Self(T::simd_le(self.0, other.0), self.1)
            }}

            /// Lane-wise greater-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> Self {{
                Self(T::simd_gt(self.0, other.0), self.1)
            }}

            /// Lane-wise greater-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> Self {{
                Self(T::simd_ge(self.0, other.0), self.1)
            }}

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            #[inline(always)]
            pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{
                Self(T::blend(mask.0, if_true.0, if_false.0), mask.1)
            }}

    "}
}

fn gen_approximations() -> String {
    formatdoc! {"
        \x20   /// Fast reciprocal approximation (~12-bit precision).
            #[inline(always)]
            pub fn rcp_approx(self) -> Self {{
                Self(T::rcp_approx(self.1, self.0), self.1)
            }}

            /// Precise reciprocal (Newton-Raphson refined).
            #[inline(always)]
            pub fn recip(self) -> Self {{
                Self(T::recip(self.1, self.0), self.1)
            }}

            /// Fast reciprocal square root approximation (~12-bit precision).
            #[inline(always)]
            pub fn rsqrt_approx(self) -> Self {{
                Self(T::rsqrt_approx(self.1, self.0), self.1)
            }}

            /// Precise reciprocal square root (Newton-Raphson refined).
            #[inline(always)]
            pub fn rsqrt(self) -> Self {{
                Self(T::rsqrt(self.1, self.0), self.1)
            }}

    "}
}

fn gen_shifts(ty: &SimdType) -> String {
    let mut code = formatdoc! {"
        \x20   /// Shift left by constant.
            #[inline(always)]
            pub fn shl_const<const N: i32>(self) -> Self {{
                Self(T::shl_const::<N>(self.0), self.1)
            }}

    "};

    if has_shr_arithmetic(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Arithmetic shift right by constant (sign-extending).
                #[inline(always)]
                pub fn shr_arithmetic_const<const N: i32>(self) -> Self {{
                    Self(T::shr_arithmetic_const::<N>(self.0), self.1)
                }}

        "});
    }

    code.push_str(&formatdoc! {"
        \x20   /// Logical shift right by constant (zero-filling).
            #[inline(always)]
            pub fn shr_logical_const<const N: i32>(self) -> Self {{
                Self(T::shr_logical_const::<N>(self.0), self.1)
            }}

            /// Alias for [`shl_const`](Self::shl_const).
            #[inline(always)]
            pub fn shl<const N: i32>(self) -> Self {{
                self.shl_const::<N>()
            }}

    "});

    if has_shr_arithmetic(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Alias for [`shr_arithmetic_const`](Self::shr_arithmetic_const).
                #[inline(always)]
                pub fn shr_arithmetic<const N: i32>(self) -> Self {{
                    self.shr_arithmetic_const::<N>()
                }}

        "});
    }

    code.push_str(&formatdoc! {"
        \x20   /// Alias for [`shr_logical_const`](Self::shr_logical_const).
            #[inline(always)]
            pub fn shr_logical<const N: i32>(self) -> Self {{
                self.shr_logical_const::<N>()
            }}

    "});

    code
}

fn gen_partition_slice(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let needs_multiline = lanes >= 16;

    if needs_multiline {
        formatdoc! {"
            \x20   /// Split a slice into SIMD-width chunks and a scalar remainder.
                ///
                /// Returns `(&[[{elem}; {lanes}]], &[{elem}])` — the bulk portion reinterpreted
                /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
                #[inline(always)]
                pub fn partition_slice(_: T, data: &[{elem}]) -> (&[[{elem}; {lanes}]], &[{elem}]) {{
                    let bulk = data.len() / {lanes};
                    let (head, tail) = data.split_at(bulk * {lanes});
                    // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                    // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                    let chunks =
                        unsafe {{ core::slice::from_raw_parts(head.as_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                    (chunks, tail)
                }}

        "}
    } else {
        formatdoc! {"
            \x20   /// Split a slice into SIMD-width chunks and a scalar remainder.
                ///
                /// Returns `(&[[{elem}; {lanes}]], &[{elem}])` — the bulk portion reinterpreted
                /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
                #[inline(always)]
                pub fn partition_slice(_: T, data: &[{elem}]) -> (&[[{elem}; {lanes}]], &[{elem}]) {{
                    let bulk = data.len() / {lanes};
                    let (head, tail) = data.split_at(bulk * {lanes});
                    // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                    // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                    let chunks = unsafe {{ core::slice::from_raw_parts(head.as_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                    (chunks, tail)
                }}

        "}
    }
}

fn gen_partition_slice_mut(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    formatdoc! {"
        \x20   /// Split a mutable slice into SIMD-width chunks and a scalar remainder.
            ///
            /// Returns `(&mut [[{elem}; {lanes}]], &mut [{elem}])` — the bulk portion reinterpreted
            /// as fixed-size arrays suitable for [`load`](Self::load), plus any leftover elements.
            #[inline(always)]
            pub fn partition_slice_mut(_: T, data: &mut [{elem}]) -> (&mut [[{elem}; {lanes}]], &mut [{elem}]) {{
                let bulk = data.len() / {lanes};
                let (head, tail) = data.split_at_mut(bulk * {lanes});
                // SAFETY: head.len() is bulk * {lanes}, so it's exactly `bulk` chunks of [{elem}; {lanes}].
                // The pointer cast is valid because [{elem}] and [[{elem}; {lanes}]] have the same alignment.
                let chunks =
                    unsafe {{ core::slice::from_raw_parts_mut(head.as_mut_ptr().cast::<[{elem}; {lanes}]>(), bulk) }};
                (chunks, tail)
            }}

    "}
}

// ============================================================================
// Operators, assign operators, scalar broadcast, index, from-array, debug
// ============================================================================

fn gen_operators(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    let mut code = formatdoc! {"
        // ============================================================================
        // Operator implementations
        // ============================================================================

    "};

    code.push_str(&gen_binary_op(&name, &backend, "Add", "add"));
    code.push_str(&gen_binary_op(&name, &backend, "Sub", "sub"));
    if has_mul(ty.elem) {
        code.push_str(&gen_binary_op(&name, &backend, "Mul", "mul"));
    }
    if has_div(ty.elem) {
        code.push_str(&gen_binary_op(&name, &backend, "Div", "div"));
    }
    if has_neg(ty.elem) {
        code.push_str(&formatdoc! {"
            impl<T: {backend}> Neg for {name}<T> {{
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {{
                    Self(T::neg(self.1, self.0), self.1)
                }}
            }}

        "});
    }
    code.push_str(&gen_binary_op(&name, &backend, "BitAnd", "bitand"));
    code.push_str(&gen_binary_op(&name, &backend, "BitOr", "bitor"));
    code.push_str(&gen_binary_op(&name, &backend, "BitXor", "bitxor"));
    code
}

fn gen_binary_op(name: &str, backend: &str, trait_name: &str, method: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name} for {name}<T> {{
            type Output = Self;
            #[inline(always)]
            fn {method}(self, rhs: Self) -> Self {{
                Self(T::{method}(self.0, rhs.0), self.1)
            }}
        }}

    "}
}

fn gen_assign_operators(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    let mut code = formatdoc! {"
        // ============================================================================
        // Assign operators
        // ============================================================================

    "};

    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "AddAssign",
        "add_assign",
        "+",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "SubAssign",
        "sub_assign",
        "-",
    ));
    if has_mul(ty.elem) {
        code.push_str(&gen_assign_op(
            &name,
            &backend,
            "MulAssign",
            "mul_assign",
            "*",
        ));
    }
    if has_div(ty.elem) {
        code.push_str(&gen_assign_op(
            &name,
            &backend,
            "DivAssign",
            "div_assign",
            "/",
        ));
    }
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitAndAssign",
        "bitand_assign",
        "&",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitOrAssign",
        "bitor_assign",
        "|",
    ));
    code.push_str(&gen_assign_op(
        &name,
        &backend,
        "BitXorAssign",
        "bitxor_assign",
        "^",
    ));
    code
}

fn gen_assign_op(name: &str, backend: &str, trait_name: &str, method: &str, op: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name} for {name}<T> {{
            #[inline(always)]
            fn {method}(&mut self, rhs: Self) {{
                *self = *self {op} rhs;
            }}
        }}

    "}
}

fn gen_scalar_broadcast(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let backend = backend_trait(ty);

    let header_comment = if ty.elem.is_float() {
        format!("Scalar broadcast operators (v + 2.0, v * 0.5, etc.)")
    } else if has_mul(ty.elem) {
        format!("Scalar broadcast operators (v + 2, v * 3, etc.)")
    } else {
        format!("Scalar broadcast operators (v + 2, etc.)")
    };

    let mut code = formatdoc! {"
        // ============================================================================
        // {header_comment}
        // ============================================================================

    "};

    code.push_str(&gen_scalar_op(&name, &backend, elem, "Add", "add"));
    code.push_str(&gen_scalar_op(&name, &backend, elem, "Sub", "sub"));
    if has_mul(ty.elem) {
        code.push_str(&gen_scalar_op(&name, &backend, elem, "Mul", "mul"));
    }
    if has_div(ty.elem) {
        code.push_str(&gen_scalar_op(&name, &backend, elem, "Div", "div"));
    }
    code
}

fn gen_scalar_op(name: &str, backend: &str, elem: &str, trait_name: &str, method: &str) -> String {
    formatdoc! {"
        impl<T: {backend}> {trait_name}<{elem}> for {name}<T> {{
            type Output = Self;
            #[inline(always)]
            fn {method}(self, rhs: {elem}) -> Self {{
                Self(T::{method}(self.0, T::splat(self.1, rhs)), self.1)
            }}
        }}

    "}
}

fn gen_index(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Index
        // ============================================================================

        impl<T: {backend}> Index<usize> for {name}<T> {{
            type Output = {elem};
            #[inline(always)]
            fn index(&self, i: usize) -> &{elem} {{
                assert!(i < {lanes}, \"{name} index out of bounds: {{i}}\");
                // SAFETY: {name}'s repr is layout-compatible with [{elem}; {lanes}], and i < {lanes}.
                unsafe {{ &*(core::ptr::from_ref(self).cast::<{elem}>()).add(i) }}
            }}
        }}

        impl<T: {backend}> IndexMut<usize> for {name}<T> {{
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut {elem} {{
                assert!(i < {lanes}, \"{name} index out of bounds: {{i}}\");
                // SAFETY: {name}'s repr is layout-compatible with [{elem}; {lanes}], and i < {lanes}.
                unsafe {{ &mut *(core::ptr::from_mut(self).cast::<{elem}>()).add(i) }}
            }}
        }}

    "}
}

fn gen_from_array(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Conversions
        // ============================================================================

        impl<T: {backend}> From<{name}<T>> for [{elem}; {lanes}] {{
            #[inline(always)]
            fn from(v: {name}<T>) -> [{elem}; {lanes}] {{
                T::to_array(v.0)
            }}
        }}

    "}
}

fn gen_debug(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    formatdoc! {"
        // ============================================================================
        // Debug
        // ============================================================================

        impl<T: {backend}> core::fmt::Debug for {name}<T> {{
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
                let arr = T::to_array(self.0);
                f.debug_tuple(\"{name}\").field(&arr).finish()
            }}
        }}

    "}
}

fn gen_cross_type(ty: &SimdType) -> String {
    let name = ty.name();
    let conversions = all_conversions();
    let matching: Vec<&Conversion> = conversions.iter().filter(|c| c.src == name).collect();

    if matching.is_empty() {
        return String::new();
    }

    let mut code = String::new();
    for conv in &matching {
        code.push_str(&(conv.gen_fn)(&name, conv.trait_bound));
        code.push('\n');
    }
    code
}

fn gen_platform(ty: &SimdType) -> String {
    let name = ty.name();

    match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => {
            let raw_type = x86_raw_type(ty);
            let from_fn = from_raw_fn_name(ty);

            formatdoc! {"
                // ============================================================================
                // Platform-specific concrete impls
                // ============================================================================

                #[cfg(target_arch = \"x86_64\")]
                impl {name}<archmage::X64V3Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v3::{name}\"
                    }}

                    /// Get the raw `{raw_type}` value.
                    #[inline(always)]
                    pub fn raw(self) -> core::arch::x86_64::{raw_type} {{
                        self.0
                    }}

                    /// Create from a raw `{raw_type}` (token-gated, zero-cost).
                    #[inline(always)]
                    pub fn {from_fn}(token: archmage::X64V3Token, v: core::arch::x86_64::{raw_type}) -> Self {{
                        Self(v, token)
                    }}
                }}
            "}
        }
        SimdWidth::W512 => {
            formatdoc! {"
                // ============================================================================
                // Platform-specific implementation info
                // ============================================================================

                #[cfg(target_arch = \"x86_64\")]
                impl {name}<archmage::X64V3Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::v3_512::{name}\"
                    }}
                }}

                #[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]
                impl {name}<archmage::X64V4Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v4::{name}\"
                    }}
                }}

                #[cfg(all(target_arch = \"x86_64\", feature = \"avx512\"))]
                impl {name}<archmage::X64V4xToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"x86::v4x::{name}\"
                    }}
                }}
            "}
        }
    }
}

fn gen_popcnt(ty: &SimdType) -> String {
    if ty.width != SimdWidth::W512 || ty.elem.is_float() {
        return String::new();
    }

    let name = ty.name();

    formatdoc! {"

        // ============================================================================
        // Extension: popcnt (requires Modern token)
        // ============================================================================

        #[cfg(feature = \"avx512\")]
        impl<T: crate::simd::backends::{name}PopcntBackend> {name}<T> {{
            /// Count set bits in each lane (popcnt).
            ///
            /// Returns a vector where each lane contains the number of 1-bits
            /// in the corresponding lane of `self`.
            ///
            /// Requires AVX-512 Modern token (VPOPCNTDQ or BITALG extension).
            #[inline(always)]
            pub fn popcnt(self) -> Self {{
                Self(T::popcnt(self.0), self.1)
            }}
        }}
    "}
}

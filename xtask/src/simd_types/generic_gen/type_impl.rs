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
    code.push_str(&conversions::gen_widen_narrow(&ty.name()));
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

    let layout_asserts = gen_layout_asserts(ty);

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
        // SAFETY: repr(C) pair of Pod storage and a sealed 1-ZST token.
        // A supplied T proves CPU support; the wrapper adds no bit invariants.
        // Helpers additionally check token size/alignment at monomorphization.
        unsafe impl<T: {backend}> crate::simd_storage::TokenStorage for {name}<T> {{
            type Token = T;
        }}
        {phantom_comment}{layout_asserts}
    "}
}

// ============================================================================
// Compile-time layout assertions
//
// Under `#[repr(C)]` with a trailing ZST token field, the struct has the same
// size and alignment as `T::Repr` *when* `T` is a 1-ZST (zero size,
// alignment of 1). All archmage tokens are currently 1-ZSTs, but that could
// change if a future refactor adds a non-ZST field to a token (e.g. by accident
// during a code cleanup). These asserts catch that regression at compile time.
//
// One assert is emitted unconditionally using `ScalarToken` (which implements
// every backend trait and is always available). Platform-native tokens get
// additional asserts gated by the matching `target_arch` (and `feature` for V4).
// ============================================================================

fn gen_layout_asserts(ty: &SimdType) -> String {
    let name = ty.name();
    let backend = backend_trait(ty);

    // ScalarToken assert — always present, works on every target.
    let scalar_block = formatdoc! {"

        // Layout invariant: struct is `#[repr(C)]` with a trailing ZST `T`
        // field, so `sizeof/alignof({name}<T>) == sizeof/alignof(T::Repr)`
        // when `T` is a 1-ZST. Every archmage token currently satisfies this;
        // if a future refactor adds a non-ZST field to a token, this const
        // assert fires at compile time.
        const _: () = {{
            assert!(
                core::mem::size_of::<{name}<archmage::ScalarToken>>()
                    == core::mem::size_of::<<archmage::ScalarToken as crate::simd::backends::{backend}>::Repr>()
            );
            assert!(
                core::mem::align_of::<{name}<archmage::ScalarToken>>()
                    == core::mem::align_of::<<archmage::ScalarToken as crate::simd::backends::{backend}>::Repr>()
            );
        }};
    "};

    // Native-token asserts gated by target_arch (and the w512/avx512 features
    // for the 512-bit native path on x86).
    let mut native_blocks = String::new();

    // x86_64: all widths use X64V3Token for its backend impl, except that
    // the W512 avx512 backend lives on X64V4Token behind a feature gate.
    // We pick X64V3Token for W128/W256 (its Repr is the native `__m128`/
    // `__m256`); for W512 we add two variants: one using V3's polyfill
    // (`[__m256; 2]`), one using V4's native `__m512` under `feature = "avx512"`.
    match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => {
            native_blocks.push_str(&formatdoc! {r#"

                #[cfg(target_arch = "x86_64")]
                const _: () = {{
                    assert!(
                        core::mem::size_of::<{name}<archmage::X64V3Token>>()
                            == core::mem::size_of::<<archmage::X64V3Token as crate::simd::backends::{backend}>::Repr>()
                    );
                    assert!(
                        core::mem::align_of::<{name}<archmage::X64V3Token>>()
                            == core::mem::align_of::<<archmage::X64V3Token as crate::simd::backends::{backend}>::Repr>()
                    );
                }};
            "#});
        }
        SimdWidth::W512 => {
            // W512 on V3 is the polyfill path (`[__m256; 2]`). The full file is
            // gated on `feature = "w512"` already (see generic/generated/mod.rs),
            // so we don't need an extra feature gate here.
            native_blocks.push_str(&formatdoc! {r#"

                #[cfg(target_arch = "x86_64")]
                const _: () = {{
                    assert!(
                        core::mem::size_of::<{name}<archmage::X64V3Token>>()
                            == core::mem::size_of::<<archmage::X64V3Token as crate::simd::backends::{backend}>::Repr>()
                    );
                    assert!(
                        core::mem::align_of::<{name}<archmage::X64V3Token>>()
                            == core::mem::align_of::<<archmage::X64V3Token as crate::simd::backends::{backend}>::Repr>()
                    );
                }};

                // Native AVX-512 (`__m512`/`__m512d`/`__m512i`) — gated on the
                // `avx512` feature, which is how archmage exposes X64V4Token's
                // 512-bit backend impls.
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                const _: () = {{
                    assert!(
                        core::mem::size_of::<{name}<archmage::X64V4Token>>()
                            == core::mem::size_of::<<archmage::X64V4Token as crate::simd::backends::{backend}>::Repr>()
                    );
                    assert!(
                        core::mem::align_of::<{name}<archmage::X64V4Token>>()
                            == core::mem::align_of::<<archmage::X64V4Token as crate::simd::backends::{backend}>::Repr>()
                    );
                }};
            "#});
        }
    }

    // aarch64 NEON — same backend trait; polyfill for wider-than-128 widths.
    native_blocks.push_str(&formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        const _: () = {{
            assert!(
                core::mem::size_of::<{name}<archmage::NeonToken>>()
                    == core::mem::size_of::<<archmage::NeonToken as crate::simd::backends::{backend}>::Repr>()
            );
            assert!(
                core::mem::align_of::<{name}<archmage::NeonToken>>()
                    == core::mem::align_of::<<archmage::NeonToken as crate::simd::backends::{backend}>::Repr>()
            );
        }};
    "#});

    // wasm32 SIMD128 — gated on `target_feature = "simd128"` because the
    // Wasm128Token only implements the backends on the +simd128 target.
    native_blocks.push_str(&formatdoc! {r#"

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        const _: () = {{
            assert!(
                core::mem::size_of::<{name}<archmage::Wasm128Token>>()
                    == core::mem::size_of::<<archmage::Wasm128Token as crate::simd::backends::{backend}>::Repr>()
            );
            assert!(
                core::mem::align_of::<{name}<archmage::Wasm128Token>>()
                    == core::mem::align_of::<<archmage::Wasm128Token as crate::simd::backends::{backend}>::Repr>()
            );
        }};
    "#});

    format!("{scalar_block}{native_blocks}")
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
                T::reduce_add(self.1, self.0)
            }}

    "});
    if has_reduce_min_max(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Minimum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_min(self) -> {elem} {{
                    T::reduce_min(self.1, self.0)
                }}

                /// Maximum across all {lanes} lanes.
                #[inline(always)]
                pub fn reduce_max(self) -> {elem} {{
                    T::reduce_max(self.1, self.0)
                }}

        "});
    }

    // ====== Approximations ======
    if has_approx(ty.elem) {
        code.push_str("    // ====== Approximations ======\n\n");
        code.push_str(&gen_approximations(ty));
    }

    // ====== Deterministic reciprocals (f32 only) ======
    if ty.elem == ElementType::F32 {
        code.push_str(&gen_deterministic_reciprocals(ty));
    }

    // ====== Shifts ======
    if has_shifts(ty.elem) {
        code.push_str("    // ====== Shifts ======\n\n");
        code.push_str(&gen_shifts(ty));
    }

    // ====== Uniform variable shifts + saturating arithmetic ======
    if has_uniform_shifts(ty.elem) {
        code.push_str(&gen_uniform_shifts(ty));
    }
    if has_saturating_arith(ty.elem) {
        code.push_str(&gen_saturating_arith(ty));
    }

    // ====== Bitwise ======
    code.push_str("    // ====== Bitwise ======\n\n");
    code.push_str(&formatdoc! {"
        \x20   /// Bitwise NOT.
            #[inline(always)]
            pub fn not(self) -> Self {{
                Self(T::not(self.1, self.0), self.1)
            }}

    "});

    // ====== Boolean ======
    if has_boolean(ty.elem) {
        code.push_str("    // ====== Boolean ======\n\n");
        code.push_str(&formatdoc! {"
            \x20   /// True if all lanes have their {high_bit} set (all-1s mask).
                #[inline(always)]
                pub fn all_true(self) -> bool {{
                    T::all_true(self.1, self.0)
                }}

                /// True if any lane has its {high_bit} set.
                #[inline(always)]
                pub fn any_true(self) -> bool {{
                    T::any_true(self.1, self.0)
                }}

                /// Extract the high bit of each {elem_bits}-bit lane as a bitmask.
                #[inline(always)]
                pub fn bitmask(self) -> {bitmask_ty} {{
                    T::bitmask(self.1, self.0)
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
                T::store(self.1, self.0, out);
            }}

            /// Convert to array.
            #[inline(always)]
            pub fn to_array(self) -> [{elem}; {lanes}] {{
                T::to_array(self.1, self.0)
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
                Self(T::min(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise maximum.
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
                Self(T::max(self.1, self.0, other.0), self.1)
            }}

            /// Clamp between lo and hi.
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                Self(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
            }}

            /// Square root.
            #[inline(always)]
            pub fn sqrt(self) -> Self {{
                Self(T::sqrt(self.1, self.0), self.1)
            }}

            /// Absolute value.
            #[inline(always)]
            pub fn abs(self) -> Self {{
                Self(T::abs(self.1, self.0), self.1)
            }}

            /// Round toward negative infinity.
            #[inline(always)]
            pub fn floor(self) -> Self {{
                Self(T::floor(self.1, self.0), self.1)
            }}

            /// Round toward positive infinity.
            #[inline(always)]
            pub fn ceil(self) -> Self {{
                Self(T::ceil(self.1, self.0), self.1)
            }}

            /// Round to nearest integer.
            #[inline(always)]
            pub fn round(self) -> Self {{
                Self(T::round(self.1, self.0), self.1)
            }}

            /// Multiply-add: `self * a + b`.
            ///
            /// Fused with a single rounding on backends with hardware FMA
            /// (x86 v3/v4, NEON). The scalar and WASM backends compute an
            /// unfused `mul` + `add` (two roundings), so lanes can differ
            /// from the fused backends by 1 ULP.
            #[inline(always)]
            pub fn mul_add(self, a: Self, b: Self) -> Self {{
                Self(T::mul_add(self.1, self.0, a.0, b.0), self.1)
            }}

            /// Multiply-sub: `self * a - b`.
            ///
            /// Same fusion contract as [`mul_add`](Self::mul_add): fused on
            /// x86 v3/v4 and NEON, unfused (two roundings) on scalar and WASM.
            #[inline(always)]
            pub fn mul_sub(self, a: Self, b: Self) -> Self {{
                Self(T::mul_sub(self.1, self.0, a.0, b.0), self.1)
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
                Self(T::min(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise maximum{unsigned_suffix}.
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
                Self(T::max(self.1, self.0, other.0), self.1)
            }}

    "};

    if has_abs(ty.elem) && !ty.elem.is_float() {
        code.push_str(&formatdoc! {"
            \x20   /// Lane-wise absolute value.
                #[inline(always)]
                pub fn abs(self) -> Self {{
                    Self(T::abs(self.1, self.0), self.1)
                }}

        "});
    }

    code.push_str(&formatdoc! {"
        \x20   /// Clamp between lo and hi.
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                Self(T::clamp(self.1, self.0, lo.0, hi.0), self.1)
            }}

    "});

    code
}

fn gen_comparisons(signedness: &str) -> String {
    formatdoc! {"
        \x20   /// Lane-wise equality (returns mask).
            #[inline(always)]
            pub fn simd_eq(self, other: Self) -> Self {{
                Self(T::simd_eq(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise inequality (returns mask).
            #[inline(always)]
            pub fn simd_ne(self, other: Self) -> Self {{
                Self(T::simd_ne(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise less-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> Self {{
                Self(T::simd_lt(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise less-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_le(self, other: Self) -> Self {{
                Self(T::simd_le(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise greater-than{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> Self {{
                Self(T::simd_gt(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise greater-than-or-equal{signedness} (returns mask).
            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> Self {{
                Self(T::simd_ge(self.1, self.0, other.0), self.1)
            }}

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            #[inline(always)]
            pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{
                Self(T::blend(mask.1, mask.0, if_true.0, if_false.0), mask.1)
            }}

    "}
}

fn gen_approximations(ty: &SimdType) -> String {
    let lanes = ty.lanes();
    if ty.elem == ElementType::F32 && lanes <= 8 {
        formatdoc! {"
            \x20   /// Fast reciprocal (1/x), ≥~12-bit floor by the cheapest path per
                /// platform: x86 ~12-bit (raw hardware estimate), ARM ~16-bit
                /// (estimate + one Newton step), WASM/scalar ~24-bit (exact division —
                /// no hardware estimate exists to undercut it). For the same bits on
                /// *every* machine use [`rcp_approx_portable`](Self::rcp_approx_portable);
                /// for ≤4 ULP use [`recip`](Self::recip); for 0 ULP + IEEE rails
                /// use [`recip_portable`](Self::recip_portable).
                ///
                /// Rails are UNSPECIFIED at this tier (the current lowerings
                /// happen to return IEEE values on x86/NEON/WASM, but only
                /// [`recip`](Self::recip) and up contract it).
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                    // Each backend owns its >=12-bit estimate (x86 raw rcpps; ARM
                    // raw vrecpe + 1 fused FRECPS; WASM/scalar exact division).
                    Self(T::rcp_approx(self.1, self.0), self.1)
                }}

                /// Reciprocal (1/x), the working tier: at least ~22 correct bits
                /// (≤4 ULP) **with exact IEEE rails** — `recip(±0) = ±inf`,
                /// `recip(±inf) = ±0` (signed), NaN propagates — on every backend.
                ///
                /// Fastest conforming path per backend: estimate + Newton + a
                /// branchless rail rescue on x86 (+2.6% over the raw Newton form,
                /// vs +21% for exact division); NEON's FRECPS special-cases
                /// `inf·0` in hardware so its refinement passes rails through for
                /// free; WASM/scalar divide (0 ULP there). The #64 footgun
                /// (`(exp_midp() + one).recip()` NaN'ing on saturated lanes)
                /// cannot occur at this tier.
                ///
                /// **Subnormal inputs are the one unspecified case** — for 0 ULP
                /// everywhere including subnormals, plus bit-identical
                /// cross-arch results, use
                /// [`recip_portable`](Self::recip_portable).
                #[inline(always)]
                pub fn recip(self) -> Self {{
                    Self(T::recip(self.1, self.0), self.1)
                }}

                /// Fast reciprocal square root (1/sqrt(x)), ≥~12-bit floor — see
                /// [`rcp_approx`](Self::rcp_approx) for the per-platform strategy.
                ///
                /// Rails are UNSPECIFIED at this tier (the scalar bit-hack in
                /// particular returns garbage at `±0`); [`rsqrt`](Self::rsqrt)
                /// and up contract them.
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                    // ARM uses raw vrsqrte + 1 fused FRSQRTS; WASM/scalar a bit-hack
                    // seed + 2 Newton steps; x86 the raw rsqrtps estimate.
                    Self(T::rsqrt_approx(self.1, self.0), self.1)
                }}

                /// Reciprocal square root (1/sqrt(x)), the working tier: ≤4 ULP
                /// **with exact IEEE rails** (`rsqrt(±0) = ±inf`,
                /// `rsqrt(+inf) = +0`, negatives and NaN give NaN) on every
                /// backend — estimate + Newton + rail rescue on x86 (still ~2.8x
                /// faster than exact), hardware-special-cased FRSQRTS on NEON
                /// (free), division on WASM/scalar. Deep-subnormal inputs
                /// unspecified; use [`rsqrt_portable`](Self::rsqrt_portable) for
                /// 0 ULP + subnormals + bit-identical.
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                    Self(T::rsqrt(self.1, self.0), self.1)
                }}

        "}
    } else if ty.elem == ElementType::F32 {
        // f32x16: same working-tier contract as the narrower f32 types
        // (estimate + Newton on x86 native 512-bit and on the delegated
        // polyfills; exact division on WASM/scalar).
        formatdoc! {"
            \x20   /// Fast reciprocal approximation (1/x): the backend's native estimate.
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                    Self(T::rcp_approx(self.1, self.0), self.1)
                }}

                /// Reciprocal (1/x), the working tier: ≤4 ULP **with exact IEEE
                /// rails** on every backend (native AVX-512 uses rcp14 + Newton +
                /// a one-instruction VFIXUPIMM rail patch; polyfill widths
                /// inherit their sub-vector's rescue). Subnormal inputs
                /// unspecified; [`recip_portable`](Self::recip_portable) for
                /// 0 ULP + subnormals + bit-identical.
                #[inline(always)]
                pub fn recip(self) -> Self {{
                    Self(T::recip(self.1, self.0), self.1)
                }}

                /// Fast reciprocal square root approximation: the backend's native
                /// estimate. `±0`/`±inf` lanes can come back NaN on estimate-refine
                /// backends.
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                    Self(T::rsqrt_approx(self.1, self.0), self.1)
                }}

                /// Reciprocal square root (1/sqrt(x)), the working tier: ≤4 ULP
                /// with exact IEEE rails — see [`recip`](Self::recip) for the
                /// contract shape; [`rsqrt_portable`](Self::rsqrt_portable) adds
                /// 0 ULP + subnormals + bit-identical.
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                    Self(T::rsqrt(self.1, self.0), self.1)
                }}

        "}
    } else {
        // f64: no estimate hardware worth refining exists at any width, so the
        // bare names are exact division everywhere — 0 ULP with IEEE rails.
        formatdoc! {"
            \x20   /// Fast reciprocal approximation (1/x): the backend's native estimate
                /// (exact division on f64 — no hardware estimate exists).
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                    Self(T::rcp_approx(self.1, self.0), self.1)
                }}

                /// Precise reciprocal (1/x): exact IEEE division on every backend.
                ///
                /// Correctly rounded (0 ULP vs `1.0 / x`), including the rails:
                /// `recip(±0) = ±inf` and `recip(±inf) = ±0` (issue #64). On f64
                /// the working tier and the exact tier coincide.
                #[inline(always)]
                pub fn recip(self) -> Self {{
                    Self(T::recip(self.1, self.0), self.1)
                }}

                /// Fast reciprocal square root approximation (exact division +
                /// sqrt on f64 — no hardware estimate exists).
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                    Self(T::rsqrt_approx(self.1, self.0), self.1)
                }}

                /// Precise reciprocal square root (1/sqrt(x)): exact IEEE division
                /// and square root on every backend. Bit-exact vs scalar
                /// `1.0 / x.sqrt()`, rails included.
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                    Self(T::rsqrt(self.1, self.0), self.1)
                }}

        "}
    }
}

/// Deterministic, cross-platform **bit-identical** reciprocal / reciprocal-sqrt
/// (f32 only). Generated for f32x4/f32x8/f32x16.
fn gen_deterministic_reciprocals(ty: &SimdType) -> String {
    let lanes = ty.lanes();
    formatdoc! {r#"
            // ====== Deterministic (cross-platform bit-identical) reciprocals ======
            //
            // The `_portable` family returns **the same bits on every architecture**
            // (x86 / ARM / WASM). Two ingredients make that hold:
            //   * the seed is an integer bit-trick — pure integer math, so it is
            //     identical by construction (the shift is logical, not arch-dependent);
            //   * the Newton steps use plain IEEE-754 `mul`/`sub`, never `mul_add`
            //     (FMA): WASM SIMD has no fused FMA, and FMA-vs-non-FMA itself
            //     diverges, so fused ops are avoided on purpose.
            //
            // Hardware estimate instructions (`rsqrtps`, `vrsqrte`) are deliberately
            // NOT used here — their bits differ across vendors and generations. For
            // the faster, per-platform (non-deterministic) variants see
            // [`rsqrt_approx`](Self::rsqrt_approx) / [`recip`](Self::recip).

            /// Deterministic reciprocal-sqrt estimate (~8-bit), bit-identical on
            /// every platform.
            ///
            /// Seed (integer bit-trick) plus one non-FMA Newton step. Opt into more
            /// accuracy with [`rsqrt_newton_portable`](Self::rsqrt_newton_portable)
            /// (one more step ≈ 16-bit), or use
            /// [`rsqrt_portable`](Self::rsqrt_portable) for full precision.
            #[inline(always)]
            pub fn rsqrt_approx_portable(self) -> Self
            where
                T: crate::simd::backends::F32x{lanes}Convert,
            {{
                let t = self.1;
                let bits = self.bitcast_to_i32();
                let seed = Self::from_i32_bitcast(
                    t,
                    super::i32x{lanes}::splat(t, 0x5f3759df_i32) - bits.shr_logical_const::<1>(),
                );
                seed.rsqrt_newton_portable(self)
            }}

            /// One deterministic Newton refinement step for `1/sqrt(x)`:
            /// `y * (1.5 - 0.5*x*y*y)`, where `self` is the current estimate `y`
            /// and `x` is the original input. Non-FMA → bit-identical everywhere.
            #[inline(always)]
            pub fn rsqrt_newton_portable(self, x: Self) -> Self {{
                let t = self.1;
                let half = Self::splat(t, 0.5);
                let three_halves = Self::splat(t, 1.5);
                self * (three_halves - half * x * self * self)
            }}

            /// Precise reciprocal square root: exact IEEE sqrt + division —
            /// **the 0 ULP tier**, with IEEE rails (`rsqrt(+0) = +inf`,
            /// `rsqrt(+inf) = +0`, negatives give NaN) and bit-identical on
            /// every arch. Costs ~3.6x the working-tier [`rsqrt`](Self::rsqrt)
            /// on Zen-class x86 (and is the faster form on Apple Silicon).
            #[inline(always)]
            pub fn rsqrt_portable(self) -> Self {{
                Self::splat(self.1, 1.0) / self.sqrt()
            }}

            /// Deterministic reciprocal estimate (~8-bit), bit-identical on every
            /// platform.
            ///
            /// Seed (integer bit-trick) plus one non-FMA Newton step. Opt into more
            /// accuracy with [`recip_newton_portable`](Self::recip_newton_portable),
            /// or use [`recip_portable`](Self::recip_portable) for full precision.
            #[inline(always)]
            pub fn rcp_approx_portable(self) -> Self
            where
                T: crate::simd::backends::F32x{lanes}Convert,
            {{
                let t = self.1;
                let bits = self.bitcast_to_i32();
                let seed = Self::from_i32_bitcast(
                    t,
                    super::i32x{lanes}::splat(t, 0x7ef127ea_i32) - bits,
                );
                seed.recip_newton_portable(self)
            }}

            /// One deterministic Newton refinement step for `1/x`: `y * (2 - x*y)`,
            /// where `self` is the current estimate `y` and `x` is the input.
            #[inline(always)]
            pub fn recip_newton_portable(self, x: Self) -> Self {{
                let t = self.1;
                let two = Self::splat(t, 2.0);
                self * (two - x * self)
            }}

            /// Precise reciprocal: exact IEEE division — **the 0 ULP tier**.
            ///
            /// Correctly rounded on every backend, with the IEEE rails
            /// (`recip(±0) = ±inf`, `recip(±inf) = ±0` — no NaN after a
            /// saturating `exp_midp`, issue #64) and, because correctly-rounded
            /// division is uniquely defined, bit-identical on every arch — the
            /// portable property is free. Costs ~1.9x the working-tier
            /// [`recip`](Self::recip) on Zen-class x86 (and is the FASTER form
            /// on Apple Silicon).
            #[inline(always)]
            pub fn recip_portable(self) -> Self {{
                Self::splat(self.1, 1.0) / self
            }}

    "#, lanes = lanes}
}

fn gen_shifts(ty: &SimdType) -> String {
    let max_sh = ty.elem.size_bytes() * 8 - 1;
    let mut code = formatdoc! {"
        \x20   /// Shift left by constant.
            ///
            /// `N` must be in `0..={max_sh}`; out-of-range `N` fails to compile,
            /// identically on every backend.
            #[inline(always)]
            pub fn shl_const<const N: i32>(self) -> Self {{
                const {{ assert!(N >= 0 && N <= {max_sh}, \"shift amount out of range\") }};
                Self(T::shl_const::<N>(self.1, self.0), self.1)
            }}

    "};

    if has_shr_arithmetic(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Arithmetic shift right by constant (sign-extending).
                ///
                /// `N` must be in `0..={max_sh}` (`N == 0` is the identity shift);
                /// out-of-range `N` fails to compile, identically on every backend.
                #[inline(always)]
                pub fn shr_arithmetic_const<const N: i32>(self) -> Self {{
                    const {{ assert!(N >= 0 && N <= {max_sh}, \"shift amount out of range\") }};
                    Self(T::shr_arithmetic_const::<N>(self.1, self.0), self.1)
                }}

        "});
    }

    code.push_str(&formatdoc! {"
        \x20   /// Logical shift right by constant (zero-filling).
            ///
            /// `N` must be in `0..={max_sh}` (`N == 0` is the identity shift);
            /// out-of-range `N` fails to compile, identically on every backend.
            #[inline(always)]
            pub fn shr_logical_const<const N: i32>(self) -> Self {{
                const {{ assert!(N >= 0 && N <= {max_sh}, \"shift amount out of range\") }};
                Self(T::shr_logical_const::<N>(self.1, self.0), self.1)
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

fn gen_uniform_shifts(ty: &SimdType) -> String {
    let bits = ty.elem.size_bytes() * 8;
    let max_sh = bits - 1;

    let mut code = formatdoc! {"
        \x20   // ====== Uniform variable shifts ======

            /// Shift left by a runtime `count`, applied identically to every lane.
            ///
            /// Unlike [`shl_const`](Self::shl_const), `count` is a runtime value.
            /// `count >= {bits}` yields all-zero lanes — the same result on every
            /// backend, by contract (see `docs/CROSS-ISA-INT-PRIMITIVES.md`).
            ///
            /// The count is *uniform*: one value for the whole vector. A per-lane
            /// variable shift is deliberately not offered — at 16-bit it needs
            /// AVX-512BW+VL, and wasm128 has no per-lane variable shift at all.
            #[inline(always)]
            pub fn shl_uniform(self, count: u32) -> Self {{
                Self(T::shl_uniform(self.1, self.0, count), self.1)
            }}

            /// Logical (zero-filling) shift right by a runtime `count`, applied
            /// identically to every lane.
            ///
            /// `count >= {bits}` yields all-zero lanes on every backend.
            #[inline(always)]
            pub fn shr_logical_uniform(self, count: u32) -> Self {{
                Self(T::shr_logical_uniform(self.1, self.0, count), self.1)
            }}

    ", bits = bits};

    if has_shr_arithmetic(ty.elem) {
        code.push_str(&formatdoc! {"
            \x20   /// Arithmetic (sign-filling) shift right by a runtime `count`,
                /// applied identically to every lane.
                ///
                /// `count >= {bits}` yields a sign fill (every lane becomes `0` or
                /// `-1`), equivalent to shifting by {max_sh}, on every backend.
                #[inline(always)]
                pub fn shr_arithmetic_uniform(self, count: u32) -> Self {{
                    Self(T::shr_arithmetic_uniform(self.1, self.0, count), self.1)
                }}

        ", bits = bits, max_sh = max_sh});
    }

    code
}

fn gen_saturating_arith(ty: &SimdType) -> String {
    let elem = ty.elem.name();

    formatdoc! {"
        \x20   // ====== Saturating arithmetic ======

            /// Lane-wise addition that clamps to the `{elem}` range instead of
            /// wrapping — `{elem}::saturating_add`, per lane.
            #[inline(always)]
            pub fn saturating_add(self, other: Self) -> Self {{
                Self(T::saturating_add(self.1, self.0, other.0), self.1)
            }}

            /// Lane-wise subtraction that clamps to the `{elem}` range instead of
            /// wrapping — `{elem}::saturating_sub`, per lane.
            #[inline(always)]
            pub fn saturating_sub(self, other: Self) -> Self {{
                Self(T::saturating_sub(self.1, self.0, other.0), self.1)
            }}

    ", elem = elem}
}

fn gen_partition_slice(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    formatdoc! {"
        \x20   /// Split a slice into SIMD-width chunks and a scalar remainder.
            ///
            /// Returns `(&[[{elem}; {lanes}]], &[{elem}])` — fixed-size arrays suitable
            /// for [`load`](Self::load), plus any leftover elements.
            #[inline(always)]
            pub fn partition_slice(_: T, data: &[{elem}]) -> (&[[{elem}; {lanes}]], &[{elem}]) {{
                data.as_chunks::<{lanes}>()
            }}

    "}
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
                data.as_chunks_mut::<{lanes}>()
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
                Self(T::{method}(self.1, self.0, rhs.0), self.1)
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
                Self(T::{method}(self.1, self.0, T::splat(self.1, rhs)), self.1)
            }}
        }}

    "}
}

// Array indexing supplies the sole bounds check. A separate formatted assert
// can put panic-formatting temporaries onto the valid dynamic-index path.
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
                &crate::simd_storage::view::<_, [{elem}; {lanes}]>(&self.0)[i]
            }}
        }}

        impl<T: {backend}> IndexMut<usize> for {name}<T> {{
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut {elem} {{
                &mut crate::simd_storage::view_mut::<_, [{elem}; {lanes}]>(&mut self.0)[i]
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
                T::to_array(v.1, v.0)
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
                let arr = T::to_array(self.1, self.0);
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

    // ScalarToken is available on every architecture; its impl is always compiled.
    // All other backends are cfg-gated on target_arch (and `feature = "avx512"` /
    // `feature = "w512"` where applicable).
    let scalar_block = formatdoc! {"
        // ============================================================================
        // Platform-specific concrete impls
        // ============================================================================

        impl {name}<archmage::ScalarToken> {{
            /// Implementation identifier for this backend.
            pub const fn implementation_name() -> &'static str {{
                \"scalar::{name}\"
            }}
        }}
    "};

    match ty.width {
        SimdWidth::W128 => {
            // W128 is native on every platform.
            let raw_type = x86_raw_type(ty);
            let from_fn = from_raw_fn_name(ty);

            formatdoc! {"
                {scalar_block}
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

                #[cfg(target_arch = \"aarch64\")]
                impl {name}<archmage::NeonToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"arm::neon::{name}\"
                    }}
                }}

                #[cfg(target_arch = \"wasm32\")]
                impl {name}<archmage::Wasm128Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"wasm::wasm128::{name}\"
                    }}
                }}
            "}
        }
        SimdWidth::W256 => {
            // W256 is native on x86 (AVX2); polyfilled on NEON and WASM via pairs of W128.
            let raw_type = x86_raw_type(ty);
            let from_fn = from_raw_fn_name(ty);

            formatdoc! {"
                {scalar_block}
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

                #[cfg(target_arch = \"aarch64\")]
                impl {name}<archmage::NeonToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::neon::{name}\"
                    }}
                }}

                #[cfg(target_arch = \"wasm32\")]
                impl {name}<archmage::Wasm128Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::wasm128::{name}\"
                    }}
                }}
            "}
        }
        SimdWidth::W512 => {
            // W512 is native on x86 with AVX-512 (V4/V4x); polyfilled elsewhere
            // (on V3 via pairs of W256, on NEON/WASM via quads of W128).
            // W512 types are gated on the `w512` cargo feature; scalar included.
            formatdoc! {"
                // ============================================================================
                // Platform-specific concrete impls
                // ============================================================================

                impl {name}<archmage::ScalarToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"scalar::{name}\"
                    }}
                }}

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

                #[cfg(target_arch = \"aarch64\")]
                impl {name}<archmage::NeonToken> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::neon_512::{name}\"
                    }}
                }}

                #[cfg(target_arch = \"wasm32\")]
                impl {name}<archmage::Wasm128Token> {{
                    /// Implementation identifier for this backend.
                    pub const fn implementation_name() -> &'static str {{
                        \"polyfill::wasm128_512::{name}\"
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
                Self(T::popcnt(self.1, self.0), self.1)
            }}
        }}
    "}
}

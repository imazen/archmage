//! Cross-type conversion generators (float↔int, signed↔unsigned bitcasts).

use indoc::formatdoc;

use super::{elem_prefix, lane_count};

// ============================================================================
// f32 <-> i32 conversions (full numeric + bitcast)
// ============================================================================

/// Generate f32->i32 conversions on the float type (f32x4, f32x8, or f32x16).
///
/// Generates: bitcast_to_i32, from_i32_bitcast, to_i32, to_i32_round, from_i32,
/// plus backward-compatible aliases. Ref/mut bitcast aliases are only generated
/// for types that have block_ops (f32x4, f32x8).
pub(crate) fn gen_f32_i32_convert_on_float(src: &str, trait_bound: &str) -> String {
    let lanes = lane_count(src);
    let int_type = format!("i32x{lanes}");

    let ref_aliases = if matches!(src, "f32x4" | "f32x8") {
        formatdoc! {r#"

            /// Alias for [`bitcast_ref_i32`](Self::bitcast_ref_i32) (from block_ops).
            #[inline(always)]
            pub fn bitcast_ref_{int_type}(&self) -> &super::{int_type}<T> {{
                self.bitcast_ref_i32()
            }}

            /// Alias for [`bitcast_mut_i32`](Self::bitcast_mut_i32) (from block_ops).
            #[inline(always)]
            pub fn bitcast_mut_{int_type}(&mut self) -> &mut super::{int_type}<T> {{
                self.bitcast_mut_i32()
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions (available when T implements conversion traits)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {int_type} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_i32(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(self.1, T::bitcast_f32_to_i32(self.1, self.0))
            }}

            /// Create from {int_type} via bitcast (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn from_i32_bitcast_t(token: T, v: super::{int_type}<T>) -> Self {{
                Self(T::bitcast_i32_to_f32(token, v.into_repr()), token)
            }}

            /// Convert to {int_type} with truncation toward zero.
            ///
            /// **Out-of-range and NaN lanes DIVERGE per backend** (issue #80,
            /// `docs/CROSS-ISA-DIVERGENCES.md` §3): x86's `cvttps` yields the
            /// `i32::MIN` integer-indefinite sentinel for overflow in BOTH
            /// directions and for NaN, while NEON/WASM/scalar saturate with
            /// NaN→0 — `3e9_f32` converts to −2147483648 on x86 and
            /// +2147483647 everywhere else. Keep lanes in range, or use
            /// [`to_i32_saturating`](Self::to_i32_saturating) for uniform
            /// semantics.
            #[inline(always)]
            pub fn to_i32(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(self.1, T::convert_f32_to_i32(self.1, self.0))
            }}

            /// Convert to {int_type} with truncation and UNIFORM saturation:
            /// out-of-range lanes clamp to `i32::MIN`/`i32::MAX` and NaN
            /// lanes become 0, identically on every backend — the semantics
            /// of Rust scalar `as`. Native on NEON/WASM/scalar; on x86 a
            /// 4-op compare/blend fixup over `cvttps` (issue #80).
            #[inline(always)]
            pub fn to_i32_saturating(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(self.1, T::convert_f32_to_i32_saturating(self.1, self.0))
            }}

            /// Convert to {int_type} with rounding to nearest.
            #[inline(always)]
            pub fn to_i32_round(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(self.1, T::convert_f32_to_i32_round(self.1, self.0))
            }}

            /// Create from {int_type} via numeric conversion.
            #[inline(always)]
            pub fn from_i32_t(token: T, v: super::{int_type}<T>) -> Self {{
                Self(T::convert_i32_to_f32(token, v.into_repr()), token)
            }}

            // ====== Backward-compatible aliases (old generated API names) ======

            /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
            #[inline(always)]
            pub fn bitcast_{int_type}(self) -> super::{int_type}<T> {{
                self.bitcast_to_i32()
            }}

            /// Alias for [`to_i32`](Self::to_i32).
            #[inline(always)]
            pub fn to_{int_type}(self) -> super::{int_type}<T> {{
                self.to_i32()
            }}

            /// Alias for [`to_i32_round`](Self::to_i32_round).
            #[inline(always)]
            pub fn to_{int_type}_round(self) -> super::{int_type}<T> {{
                self.to_i32_round()
            }}

            /// Alias for [`from_i32_t`](Self::from_i32_t).
            #[inline(always)]
            pub fn from_{int_type}_t(token: T, v: super::{int_type}<T>) -> Self {{
                Self::from_i32_t(token, v)
            }}
            {ref_aliases}
        }}
    "#}
}

/// Generate i32->f32 conversions on the int type (i32x4, i32x8, or i32x16).
pub(crate) fn gen_f32_i32_convert_on_int(src: &str, trait_bound: &str) -> String {
    let lanes = lane_count(src);
    let float_type = format!("f32x{lanes}");

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions (available when T implements conversion traits)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {float_type} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_f32(self) -> super::{float_type}<T> {{
                super::{float_type}::from_repr_unchecked(self.1, T::bitcast_i32_to_f32(self.1, self.0))
            }}

            /// Convert to {float_type} (numeric conversion).
            #[inline(always)]
            pub fn to_f32(self) -> super::{float_type}<T> {{
                super::{float_type}::from_repr_unchecked(self.1, T::convert_i32_to_f32(self.1, self.0))
            }}

            // ====== Backward-compatible aliases (old generated API names) ======

            /// Alias for [`bitcast_to_f32`](Self::bitcast_to_f32).
            #[inline(always)]
            pub fn bitcast_{float_type}(self) -> super::{float_type}<T> {{
                self.bitcast_to_f32()
            }}

            /// Alias for [`to_f32`](Self::to_f32).
            #[inline(always)]
            pub fn to_{float_type}(self) -> super::{float_type}<T> {{
                self.to_f32()
            }}
        }}
    "#}
}

// ============================================================================
// Unified Bitcast Generator (i8↔u8, i16↔u16, u32→i32, u64→i64, i64→f64)
// ============================================================================

/// Generate a cross-type bitcast impl block (`bitcast_*`, `bitcast_ref_*`, `bitcast_mut_*`).
///
/// When `primary_name` is `Some("bitcast_to_i32")` (or `"bitcast_to_f64"`), emits
/// that as the primary method and `bitcast_{target}` as a backward-compatible alias.
pub(crate) fn gen_bitcast(
    src: &str,
    trait_bound: &str,
    target: &str,
    backend_method: &str,
    primary_name: Option<&str>,
    swap_header_order: bool,
) -> String {
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let (h_a, h_b) = if swap_header_order {
        (target_elem, src_elem)
    } else {
        (src_elem, target_elem)
    };
    let (main_fn, alias_block) = match primary_name {
        Some(primary) => (
            primary.to_string(),
            formatdoc! {r#"


                // ====== Backward-compatible aliases ======

                /// Alias for [`{primary}`](Self::{primary}).
                #[inline(always)]
                pub fn bitcast_{target}(self) -> super::{target}<T> {{
                    self.{primary}()
                }}
            "#},
        ),
        None => (format!("bitcast_{target}"), String::new()),
    };

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({h_a} ↔ {h_b} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn {main_fn}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(self.1, T::bitcast_{backend_method}(self.1, self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                crate::simd_storage::vector_view(self.1, &self.0)
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                crate::simd_storage::vector_view_mut(self.1, &mut self.0)
            }}{alias_block}
        }}
    "#}
}

/// W512 signed/unsigned storage has identical lane bits and layout on every
/// sealed backend. Reuse the checked storage operations instead of introducing
/// per-ISA bitcast traits: no arithmetic or new unsafe boundary is needed.
pub(crate) fn gen_w512_i16_bitcast(src: &str, trait_bound: &str) -> String {
    let target = if src == "u16x32" { "i16x32" } else { "u16x32" };
    formatdoc! {r#"
        impl<T: {trait_bound}> {src}<T> {{
            /// Reinterpret all 32 lanes as {target}, preserving every bit.
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(self.1, crate::simd_storage::cast(self.0))
            }}

        }}
    "#}
}

// ============================================================================
// Widening / saturating narrowing
// ============================================================================

/// Generate the `widen_low`/`widen_high` and `narrow_saturating_*` impl blocks
/// that apply to `type_name`, if any.
pub(crate) fn gen_widen_narrow(type_name: &str) -> String {
    use crate::simd_types::backend_gen_widen_narrow::{Half, all_narrow_pairs, all_widen_pairs};

    let mut code = String::new();

    for p in all_widen_pairs().into_iter().filter(|p| p.src == type_name) {
        let trait_bound = p.src_backend();
        let dst_bound = p.dst_backend();
        let (src, dst, de) = (p.src, p.dst, p.dst_elem);
        let half = p.src_lanes / 2;
        let extend = if p.signed { "Sign" } else { "Zero" };
        code.push_str(&formatdoc! {r#"
            // ============================================================================
            // Widening ({src} -> {dst})
            // ============================================================================

            impl<T: crate::simd::backends::{trait_bound} + crate::simd::backends::{dst_bound}> {src}<T> {{
                /// {extend}-extend the low half of the lanes to `{dst}`.
                ///
                /// Result lane `i` is `self[i] as {de}` for `i` in `0..{half}`.
                /// Natural lane order on every backend. Instruction count depends
                /// on the ISA, vector width, and surrounding loads.
                #[inline(always)]
                pub fn widen_low(self) -> super::{dst}<T> {{
                    super::{dst}::from_repr_unchecked(self.1, <T as crate::simd::backends::{trait_bound}>::{lo}(self.1, self.0))
                }}

                /// {extend}-extend the high half of the lanes to `{dst}`.
                ///
                /// Result lane `i` is `self[i + {half}] as {de}`.
                #[inline(always)]
                pub fn widen_high(self) -> super::{dst}<T> {{
                    super::{dst}::from_repr_unchecked(self.1, <T as crate::simd::backends::{trait_bound}>::{hi}(self.1, self.0))
                }}
            }}

        "#, lo = p.method(Half::Low), hi = p.method(Half::High)});
    }

    for p in all_narrow_pairs()
        .into_iter()
        .filter(|p| p.src == type_name)
    {
        let trait_bound = p.src_backend();
        let sdst_bound = p.sdst_backend();
        let udst_bound = p.udst_backend();
        let src = p.src;
        let n = p.src_lanes;
        let (sdst, sde, udst, ude, se) = (p.sdst, p.sdst_elem, p.udst, p.udst_elem, p.src_elem);
        code.push_str(&formatdoc! {r#"
            // ============================================================================
            // Saturating narrowing ({src} -> {sdst} / {udst})
            // ============================================================================

            impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
                /// Narrow `self` and `high` to `{sdst}`, clamping each lane to
                /// the `{sde}` range.
                ///
                /// Result lane `i` is `self[i]` clamped for `i < {n}`, and
                /// `high[i - {n}]` clamped for `i >= {n}` — the same lane order
                /// on every backend (the AVX2 arm pays one
                /// `permute4x64` to get there).
                #[inline(always)]
                pub fn narrow_saturating_{sde}(self, high: Self) -> super::{sdst}<T>
                where T: crate::simd::backends::{sdst_bound} {{
                    super::{sdst}::from_repr_unchecked(self.1, <T as crate::simd::backends::{trait_bound}>::{sm}(self.1, self.0, high.0))
                }}

                /// Narrow `self` and `high` to `{udst}`, clamping each lane to
                /// the `{ude}` range.
                ///
                /// The source stays `{se}` to match native signed-source packs on
                /// x86 and WASM. An unsigned-source operation would require a
                /// different lowering to preserve its full input range.
                #[inline(always)]
                pub fn narrow_saturating_{ude}(self, high: Self) -> super::{udst}<T>
                where T: crate::simd::backends::{udst_bound} {{
                    super::{udst}::from_repr_unchecked(self.1, <T as crate::simd::backends::{trait_bound}>::{um}(self.1, self.0, high.0))
                }}
            }}

        "#,
            sm = p.method(true),
            um = p.method(false),
        });
    }

    code.push_str(&crate::simd_types::backend_gen_integer_ops::generic(
        type_name,
    ));
    code
}

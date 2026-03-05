//! Cross-type conversion generators (float↔int, signed↔unsigned bitcasts).

use indoc::formatdoc;

use super::{elem_prefix, lane_count, uppercase_first, x86_int_type_for_name};

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

    // block_ops ref/mut aliases only exist for types with block_ops files
    let has_block_ops = matches!(src, "f32x4" | "f32x8");
    let ref_aliases = if has_block_ops {
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
                super::{int_type}::from_repr_unchecked(T::bitcast_f32_to_i32(self.0))
            }}

            /// Create from {int_type} via bitcast (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn from_i32_bitcast(_: T, v: super::{int_type}<T>) -> Self {{
                Self(T::bitcast_i32_to_f32(v.into_repr()), PhantomData)
            }}

            /// Convert to {int_type} with truncation toward zero.
            #[inline(always)]
            pub fn to_i32(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(T::convert_f32_to_i32(self.0))
            }}

            /// Convert to {int_type} with rounding to nearest.
            #[inline(always)]
            pub fn to_i32_round(self) -> super::{int_type}<T> {{
                super::{int_type}::from_repr_unchecked(T::convert_f32_to_i32_round(self.0))
            }}

            /// Create from {int_type} via numeric conversion.
            #[inline(always)]
            pub fn from_i32(_: T, v: super::{int_type}<T>) -> Self {{
                Self(T::convert_i32_to_f32(v.into_repr()), PhantomData)
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

            /// Alias for [`from_i32`](Self::from_i32).
            #[inline(always)]
            pub fn from_{int_type}(token: T, v: super::{int_type}<T>) -> Self {{
                Self::from_i32(token, v)
            }}
            {ref_aliases}
        }}
    "#}
}

/// Generate i32->f32 conversions on the int type (i32x4 or i32x8).
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
                super::{float_type}::from_repr_unchecked(T::bitcast_i32_to_f32(self.0))
            }}

            /// Convert to {float_type} (numeric conversion).
            #[inline(always)]
            pub fn to_f32(self) -> super::{float_type}<T> {{
                super::{float_type}::from_repr_unchecked(T::convert_i32_to_f32(self.0))
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
// Signed <-> Unsigned integer bitcasts (i8/u8, i16/u16)
// ============================================================================

/// Generate signed->unsigned bitcast on the signed type (e.g., i8x16->u8x16).
pub(crate) fn gen_signed_unsigned_bitcast(
    src: &str,
    target: &str,
    to_method: &str,
    _from_method: &str,
) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{to_method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

/// Generate unsigned->signed bitcast on the unsigned type (e.g., u8x16->i8x16).
pub(crate) fn gen_unsigned_signed_bitcast(
    src: &str,
    target: &str,
    from_method: &str,
    _alias_name: &str,
) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(target));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{from_method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

// ============================================================================
// u32 -> i32 bitcasts
// ============================================================================

/// Generate u32->i32 bitcast (e.g., u32x4->i32x4, u32x8->i32x8).
pub(crate) fn gen_u32_i32_bitcast(src: &str, target: &str, method: &str) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({target_elem} ↔ {src_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_i32(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}

            // ====== Backward-compatible aliases ======

            /// Alias for [`bitcast_to_i32`](Self::bitcast_to_i32).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                self.bitcast_to_i32()
            }}
        }}
    "#}
}

// ============================================================================
// u64 -> i64 bitcasts
// ============================================================================

/// Generate u64->i64 bitcast (e.g., u64x2->i64x2, u64x4->i64x4).
pub(crate) fn gen_u64_i64_bitcast(src: &str, target: &str, method: &str) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);
    let x86_int = x86_int_type_for_name(src);

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({x86_int} / [{src_elem};{src_lanes}] / etc.)
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}
        }}
    "#}
}

// ============================================================================
// i64 -> f64 bitcasts
// ============================================================================

/// Generate i64->f64 bitcast (e.g., i64x2->f64x2, i64x4->f64x4).
pub(crate) fn gen_i64_f64_bitcast(
    src: &str,
    target: &str,
    method: &str,
    _has_alias: bool,
) -> String {
    let trait_bound = format!("{}Bitcast", uppercase_first(src));
    let src_elem = elem_prefix(src);
    let target_elem = elem_prefix(target);
    let src_lanes = lane_count(src);

    let repr_hint = if src_lanes == "2" {
        format!("__m128i/__m128d / [{src_elem};2] / etc.")
    } else {
        format!("__m256i/__m256d / [{src_elem};{src_lanes}] / etc.")
    };

    formatdoc! {r#"
        // ============================================================================
        // Cross-type conversions ({src_elem} ↔ {target_elem} bitcast)
        // ============================================================================

        impl<T: crate::simd::backends::{trait_bound}> {src}<T> {{
            /// Bitcast to {target} (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_to_f64(self) -> super::{target}<T> {{
                super::{target}::from_repr_unchecked(T::bitcast_{method}(self.0))
            }}

            /// Bitcast to {target} by reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_ref_{target}(&self) -> &super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr ({repr_hint})
                unsafe {{ &*(core::ptr::from_ref(self).cast()) }}
            }}

            /// Bitcast to {target} by mutable reference (zero-cost).
            #[inline(always)]
            pub fn bitcast_mut_{target}(&mut self) -> &mut super::{target}<T> {{
                // SAFETY: {src} and {target} share the same repr
                unsafe {{ &mut *(core::ptr::from_mut(self).cast()) }}
            }}

            // ====== Backward-compatible aliases ======

            /// Alias for [`bitcast_to_f64`](Self::bitcast_to_f64).
            #[inline(always)]
            pub fn bitcast_{target}(self) -> super::{target}<T> {{
                self.bitcast_to_f64()
            }}
        }}
    "#}
}

//! Block ops generators (array/byte views, slice casting, interleave, transpose, RGBA).

use indoc::formatdoc;

use super::backend_trait;
use crate::simd_types::types::SimdType;

pub(super) fn gen_block_ops(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let byte_size = ty.elem.size_bytes() * lanes;
    let trait_name = backend_trait(ty);

    let mut code = String::new();

    // File header
    match name.as_str() {
        "f32x4" | "f32x8" => {
            code.push_str(&formatdoc! {r#"
                //! Block, view, and image operations for `{name}<T>`.
                //!
                //! Array/byte views, slice casting, interleave/deinterleave,
                //! matrix transpose, RGBA pixel operations, and cross-type bitcast references.
            "#});
        }
        "u32x4" => {
            code.push_str(&formatdoc! {r#"
                //! Block and view operations for `{name}<T>`.
                //!
                //! Array/byte views, slice casting, and cross-type bitcast to f32x4.
            "#});
        }
        _ => {
            code.push_str(&formatdoc! {r#"
                //! Block and view operations for `{name}<T>`.
                //!
                //! Array/byte views and slice casting.
            "#});
        }
    }

    // Imports
    code.push_str(&formatdoc! {r#"

        use crate::simd::backends::{trait_name};
        use crate::simd::generic::{name};

    "#});

    // Main impl block: basic block ops (opens impl block, does NOT close it)
    code.push_str(&gen_basic_block_ops(
        &name,
        elem,
        lanes,
        byte_size,
        &trait_name,
    ));

    // Type-specific extras (f32x4/f32x8 extras close the impl block)
    match name.as_str() {
        "f32x4" => {
            code.push_str(&gen_f32x4_extras());
            code.push_str(&gen_f32_bitcast_ref_i32(&name, lanes));
        }
        "f32x8" => {
            code.push_str(&gen_f32x8_extras());
            code.push_str(&gen_f32_bitcast_ref_i32(&name, lanes));
        }
        "u32x4" => {
            // Close the basic impl block, then add separate cross-type impl block
            code.push_str("}\n");
            code.push_str(&gen_u32x4_bitcast_f32());
        }
        _ => {
            // Close the basic impl block
            code.push_str("}\n");
        }
    }

    code
}

/// Generate the basic block ops impl block (all types get these).
fn gen_basic_block_ops(
    name: &str,
    elem: &str,
    lanes: usize,
    byte_size: usize,
    trait_name: &str,
) -> String {
    formatdoc! {r#"
        impl<T: {trait_name}> {name}<T> {{
            // ====== Array/Byte Views ======

            /// Reference to underlying array (zero-copy).
            #[inline(always)]
            pub fn as_array(&self) -> &[{elem}; {lanes}] {{
                // SAFETY: {name}<T> is repr(transparent) over T::Repr, layout-compatible with [{elem}; {lanes}]
                unsafe {{ &*core::ptr::from_ref(self).cast::<[{elem}; {lanes}]>() }}
            }}

            /// Mutable reference to underlying array (zero-copy).
            #[inline(always)]
            pub fn as_array_mut(&mut self) -> &mut [{elem}; {lanes}] {{
                // SAFETY: same layout guarantee as as_array
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<[{elem}; {lanes}]>() }}
            }}

            /// View as byte array.
            #[inline(always)]
            pub fn as_bytes(&self) -> &[u8; {byte_size}] {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes for all backends
                unsafe {{ &*core::ptr::from_ref(self).cast::<[u8; {byte_size}]>() }}
            }}

            /// View as mutable byte array.
            #[inline(always)]
            pub fn as_bytes_mut(&mut self) -> &mut [u8; {byte_size}] {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes for all backends
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<[u8; {byte_size}]>() }}
            }}

            /// Create from byte array reference (token-gated).
            #[inline(always)]
            pub fn from_bytes(_: T, bytes: &[u8; {byte_size}]) -> Self {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes; transmute_copy uses read_unaligned
                unsafe {{ core::mem::transmute_copy(bytes) }}
            }}

            /// Create from owned byte array (token-gated).
            #[inline(always)]
            pub fn from_bytes_owned(_: T, bytes: [u8; {byte_size}]) -> Self {{
                // SAFETY: {name}<T> is exactly {byte_size} bytes; transmute_copy uses read_unaligned
                unsafe {{ core::mem::transmute_copy(&bytes) }}
            }}

            // ====== Slice Casting ======

            /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
            ///
            /// Returns `None` if length is not a multiple of {lanes} or alignment is wrong.
            #[inline(always)]
            pub fn cast_slice(_: T, slice: &[{elem}]) -> Option<&[Self]> {{
                if !slice.len().is_multiple_of({lanes}) {{
                    return None;
                }}
                let ptr = slice.as_ptr();
                if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
                    return None;
                }}
                let len = slice.len() / {lanes};
                // SAFETY: alignment and length checked, layout is compatible
                Some(unsafe {{ core::slice::from_raw_parts(ptr.cast::<Self>(), len) }})
            }}

            /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
            ///
            /// Returns `None` if length is not a multiple of {lanes} or alignment is wrong.
            #[inline(always)]
            pub fn cast_slice_mut(_: T, slice: &mut [{elem}]) -> Option<&mut [Self]> {{
                if !slice.len().is_multiple_of({lanes}) {{
                    return None;
                }}
                let ptr = slice.as_mut_ptr();
                if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
                    return None;
                }}
                let len = slice.len() / {lanes};
                // SAFETY: alignment and length checked, layout is compatible
                Some(unsafe {{ core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) }})
            }}
    "#}
}

/// Generate f32x4-specific extras (u8 conversions, interleave, deinterleave, rgba, transpose).
/// Returns the closing brace of the main impl block plus the extra methods.
fn gen_f32x4_extras() -> String {
    formatdoc! {r#"

            // ====== u8 Conversions ======

            /// Load 4 u8 values and convert to f32x4.
            ///
            /// Values are in `[0.0, 255.0]`. Useful for image processing.
            #[inline(always)]
            pub fn from_u8(bytes: &[u8; 4]) -> Self {{
                Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
            }}

            /// Convert to 4 u8 values with saturation.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            #[inline(always)]
            pub fn to_u8(self) -> [u8; 4] {{
                let arr = self.to_array();
                core::array::from_fn(|i| crate::nostd_math::roundf(arr[i]).clamp(0.0, 255.0) as u8)
            }}

            // ====== Interleave Operations ======

            /// Interleave low elements.
            ///
            /// ```text
            /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
            /// ```
            #[inline(always)]
            pub fn interleave_lo(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]]))
            }}

            /// Interleave high elements.
            ///
            /// ```text
            /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
            /// ```
            #[inline(always)]
            pub fn interleave_hi(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]]))
            }}

            /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
            #[inline(always)]
            pub fn interleave(self, other: Self) -> (Self, Self) {{
                let a = self.to_array();
                let b = other.to_array();
                (
                    Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]])),
                    Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]])),
                )
            }}

            // ====== 4-Channel Interleave/Deinterleave ======

            /// Deinterleave 4 RGBA pixels from AoS to SoA format.
            ///
            /// Input: 4 vectors, each containing one pixel `[R, G, B, A]`.
            /// Output: 4 vectors, each containing one channel across all pixels.
            ///
            /// This is equivalent to `transpose_4x4_copy`.
            #[inline(always)]
            pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                Self::transpose_4x4_copy(rgba)
            }}

            /// Interleave 4 channels from SoA to AoS format.
            ///
            /// Input: 4 vectors, each containing one channel across pixels.
            /// Output: 4 vectors, each containing one complete RGBA pixel.
            ///
            /// This is the inverse of `deinterleave_4ch` (also equivalent to transpose).
            #[inline(always)]
            pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                Self::transpose_4x4_copy(channels)
            }}

            // ====== RGBA Load/Store ======

            /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
            ///
            /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
            /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
            #[inline(always)]
            pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {{
                let r: [f32; 4] = core::array::from_fn(|i| rgba[i * 4] as f32);
                let g: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
                let b: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
                let a: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
                (
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                )
            }}

            /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
            #[inline(always)]
            pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {{
                let rv = r.to_array();
                let gv = g.to_array();
                let bv = b.to_array();
                let av = a.to_array();
                let mut out = [0u8; 16];
                for i in 0..4 {{
                    out[i * 4] = crate::nostd_math::roundf(rv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 1] = crate::nostd_math::roundf(gv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 2] = crate::nostd_math::roundf(bv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 3] = crate::nostd_math::roundf(av[i]).clamp(0.0, 255.0) as u8;
                }}
                out
            }}

            // ====== Matrix Transpose ======

            /// Transpose a 4x4 matrix represented as 4 row vectors (in-place).
            ///
            /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
            #[inline(always)]
            pub fn transpose_4x4(rows: &mut [Self; 4]) {{
                let a = rows[0].to_array();
                let b = rows[1].to_array();
                let c = rows[2].to_array();
                let d = rows[3].to_array();
                rows[0] = Self::from_repr_unchecked(T::from_array([a[0], b[0], c[0], d[0]]));
                rows[1] = Self::from_repr_unchecked(T::from_array([a[1], b[1], c[1], d[1]]));
                rows[2] = Self::from_repr_unchecked(T::from_array([a[2], b[2], c[2], d[2]]));
                rows[3] = Self::from_repr_unchecked(T::from_array([a[3], b[3], c[3], d[3]]));
            }}

            /// Transpose a 4x4 matrix, returning the transposed rows.
            #[inline(always)]
            pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {{
                let mut result = rows;
                Self::transpose_4x4(&mut result);
                result
            }}
        }}
    "#}
}

/// Generate f32x8-specific extras (u8 conversions, interleave, deinterleave, rgba, transpose,
/// load_8x8, store_8x8).
/// Returns the closing brace of the main impl block plus the extra methods.
fn gen_f32x8_extras() -> String {
    formatdoc! {r#"

            // ====== u8 Conversions ======

            /// Load 8 u8 values and convert to f32x8.
            ///
            /// Values are in `[0.0, 255.0]`. Useful for image processing.
            #[inline(always)]
            pub fn from_u8(bytes: &[u8; 8]) -> Self {{
                Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
            }}

            /// Convert to 8 u8 values with saturation.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            #[inline(always)]
            pub fn to_u8(self) -> [u8; 8] {{
                let arr = self.to_array();
                core::array::from_fn(|i| crate::nostd_math::roundf(arr[i]).clamp(0.0, 255.0) as u8)
            }}

            // ====== Interleave Operations ======

            /// Interleave low elements within 128-bit lanes.
            ///
            /// ```text
            /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
            /// → [a0,b0,a1,b1,a4,b4,a5,b5]
            /// ```
            #[inline(always)]
            pub fn interleave_lo(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([
                    a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5],
                ]))
            }}

            /// Interleave high elements within 128-bit lanes.
            ///
            /// ```text
            /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
            /// → [a2,b2,a3,b3,a6,b6,a7,b7]
            /// ```
            #[inline(always)]
            pub fn interleave_hi(self, other: Self) -> Self {{
                let a = self.to_array();
                let b = other.to_array();
                Self::from_repr_unchecked(T::from_array([
                    a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7],
                ]))
            }}

            /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
            #[inline(always)]
            pub fn interleave(self, other: Self) -> (Self, Self) {{
                let a = self.to_array();
                let b = other.to_array();
                (
                    Self::from_repr_unchecked(T::from_array([
                        a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5],
                    ])),
                    Self::from_repr_unchecked(T::from_array([
                        a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7],
                    ])),
                )
            }}

            // ====== 4-Channel Interleave/Deinterleave ======

            /// Deinterleave 8 RGBA pixels from AoS to SoA format.
            ///
            /// Input: 4 f32x8 vectors, each containing 2 RGBA pixels:
            /// - `rgba[0]` = `[R0, G0, B0, A0, R1, G1, B1, A1]`
            /// - `rgba[1]` = `[R2, G2, B2, A2, R3, G3, B3, A3]`
            /// - `rgba[2]` = `[R4, G4, B4, A4, R5, G5, B5, A5]`
            /// - `rgba[3]` = `[R6, G6, B6, A6, R7, G7, B7, A7]`
            ///
            /// Output: `[R_all, G_all, B_all, A_all]` — one f32x8 per channel.
            #[inline(always)]
            pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {{
                let v: [[f32; 8]; 4] = core::array::from_fn(|i| rgba[i].to_array());
                // Each input vector has 2 RGBA pixels (4 elements each)
                // Pixel i: v[i/2][(i%2)*4 + channel]
                let r: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4]);
                let g: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 1]);
                let b: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 2]);
                let a: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 3]);
                [
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                ]
            }}

            /// Interleave 4 channels from SoA to AoS format.
            ///
            /// Input: `[R, G, B, A]` — one f32x8 per channel.
            /// Output: 4 f32x8 vectors in interleaved AoS format.
            ///
            /// This is the inverse of `deinterleave_4ch`.
            #[inline(always)]
            pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {{
                let r = channels[0].to_array();
                let g = channels[1].to_array();
                let b = channels[2].to_array();
                let a = channels[3].to_array();
                core::array::from_fn(|i| {{
                    let base = i * 2;
                    Self::from_repr_unchecked(T::from_array([
                        r[base],
                        g[base],
                        b[base],
                        a[base],
                        r[base + 1],
                        g[base + 1],
                        b[base + 1],
                        a[base + 1],
                    ]))
                }})
            }}

            // ====== RGBA Load/Store ======

            /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
            ///
            /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
            /// Output: `(R, G, B, A)` where each is f32x8 with values in `[0.0, 255.0]`.
            #[inline(always)]
            pub fn load_8_rgba_u8(rgba: &[u8; 32]) -> (Self, Self, Self, Self) {{
                let r: [f32; 8] = core::array::from_fn(|i| rgba[i * 4] as f32);
                let g: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
                let b: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
                let a: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
                (
                    Self::from_repr_unchecked(T::from_array(r)),
                    Self::from_repr_unchecked(T::from_array(g)),
                    Self::from_repr_unchecked(T::from_array(b)),
                    Self::from_repr_unchecked(T::from_array(a)),
                )
            }}

            /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
            ///
            /// Values are rounded and clamped to `[0, 255]`.
            /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
            #[inline(always)]
            pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {{
                let rv = r.to_array();
                let gv = g.to_array();
                let bv = b.to_array();
                let av = a.to_array();
                let mut out = [0u8; 32];
                for i in 0..8 {{
                    out[i * 4] = crate::nostd_math::roundf(rv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 1] = crate::nostd_math::roundf(gv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 2] = crate::nostd_math::roundf(bv[i]).clamp(0.0, 255.0) as u8;
                    out[i * 4 + 3] = crate::nostd_math::roundf(av[i]).clamp(0.0, 255.0) as u8;
                }}
                out
            }}

            // ====== Matrix Transpose ======

            /// Transpose an 8x8 matrix represented as 8 row vectors (in-place).
            ///
            /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
            #[inline(always)]
            pub fn transpose_8x8(rows: &mut [Self; 8]) {{
                let r: [[f32; 8]; 8] = core::array::from_fn(|i| rows[i].to_array());
                for i in 0..8 {{
                    rows[i] = Self::from_repr_unchecked(T::from_array(core::array::from_fn(|j| r[j][i])));
                }}
            }}

            /// Transpose an 8x8 matrix, returning the transposed rows.
            #[inline(always)]
            pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {{
                let mut result = rows;
                Self::transpose_8x8(&mut result);
                result
            }}

            /// Load an 8x8 f32 block from a contiguous array into 8 row vectors.
            #[inline(always)]
            pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {{
                core::array::from_fn(|i| {{
                    let arr: [f32; 8] = block[i * 8..][..8].try_into().unwrap();
                    Self::from_repr_unchecked(T::from_array(arr))
                }})
            }}

            /// Store 8 row vectors to a contiguous 8x8 f32 block.
            #[inline(always)]
            pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {{
                for i in 0..8 {{
                    let arr = rows[i].to_array();
                    block[i * 8..][..8].copy_from_slice(&arr);
                }}
            }}
        }}
    "#}
}

/// Generate the cross-type bitcast ref/mut impl block for f32x4/f32x8 -> i32x4/i32x8.
fn gen_f32_bitcast_ref_i32(name: &str, lanes: usize) -> String {
    let int_type = format!("i32x{lanes}");
    let convert_trait = format!("F32x{}Convert", lanes);

    formatdoc! {r#"

        // ============================================================================
        // Cross-type bitcast references (require {convert_trait})
        // ============================================================================

        impl<T: crate::simd::backends::{convert_trait}> {name}<T> {{
            /// Reinterpret bits as `&{int_type}<T>` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_ref_i32(&self) -> &super::{int_type}<T> {{
                // SAFETY: {name}<T> and {int_type}<T> are both repr(transparent) with same size
                unsafe {{ &*core::ptr::from_ref(self).cast::<super::{int_type}<T>>() }}
            }}

            /// Reinterpret bits as `&mut {int_type}<T>` (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_mut_i32(&mut self) -> &mut super::{int_type}<T> {{
                // SAFETY: {name}<T> and {int_type}<T> are both repr(transparent) with same size
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<super::{int_type}<T>>() }}
            }}
        }}
    "#}
}

/// Generate the cross-type bitcast impl block for u32x4 -> f32x4.
fn gen_u32x4_bitcast_f32() -> String {
    formatdoc! {r#"

        // ============================================================================
        // Cross-type bitcast to f32x4 (requires F32x4Backend)
        // ============================================================================

        impl<T: U32x4Backend + crate::simd::backends::F32x4Backend> u32x4<T> {{
            /// Bitcast to f32x4 (reinterpret bits, no conversion).
            #[inline(always)]
            pub fn bitcast_f32x4(self) -> super::f32x4<T> {{
                // SAFETY: u32x4<T> and f32x4<T> are both exactly 16 bytes
                unsafe {{ core::mem::transmute_copy(&self) }}
            }}

            /// Bitcast to f32x4 by reference (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_ref_f32x4(&self) -> &super::f32x4<T> {{
                // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
                unsafe {{ &*core::ptr::from_ref(self).cast::<super::f32x4<T>>() }}
            }}

            /// Bitcast to f32x4 by mutable reference (zero-cost pointer cast).
            #[inline(always)]
            pub fn bitcast_mut_f32x4(&mut self) -> &mut super::f32x4<T> {{
                // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
                unsafe {{ &mut *core::ptr::from_mut(self).cast::<super::f32x4<T>>() }}
            }}
        }}
    "#}
}

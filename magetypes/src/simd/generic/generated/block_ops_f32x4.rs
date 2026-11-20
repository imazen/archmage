//! Block, view, and image operations for `f32x4<T>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x4Backend;
use crate::simd::generic::f32x4;

impl<T: F32x4Backend> f32x4<T> {
    // ====== Array/Byte Views ======
    //
    // Views borrow only raw storage. Checked helpers enforce size and
    // alignment without exposing or manufacturing the token.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        crate::simd_storage::view(&self.0)
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        crate::simd_storage::view_mut(&mut self.0)
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        crate::simd_storage::view(&self.0)
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        crate::simd_storage::view_mut(&mut self.0)
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes_t(token: T, bytes: &[u8; 16]) -> Self {
        Self(crate::simd_storage::copy(bytes), token)
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned_t(token: T, bytes: [u8; 16]) -> Self {
        Self(crate::simd_storage::cast(bytes), token)
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_t(token: T, slice: &[f32]) -> Option<&[Self]> {
        crate::simd_storage::vector_slice::<_, Self, 4>(token, slice)
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut_t(token: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        crate::simd_storage::vector_slice_mut::<_, Self, 4>(token, slice)
    }

    // ====== u8 Conversions ======

    /// Load 4 u8 values and convert to f32x4 (token-gated).
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8_t(token: T, bytes: &[u8; 4]) -> Self {
        Self::from_repr_unchecked(
            token,
            T::from_array(token, core::array::from_fn(|i| bytes[i] as f32)),
        )
    }

    /// Convert to 4 u8 values with saturation.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 4] {
        T::to_u8_bytes(self.1, self.0)
    }

    // ====== Interleave Operations ======

    /// Interleave low elements.
    ///
    /// ```text
    /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
    /// ```
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(token, T::from_array(token, [a[0], b[0], a[1], b[1]]))
    }

    /// Interleave high elements.
    ///
    /// ```text
    /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
    /// ```
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(token, T::from_array(token, [a[2], b[2], a[3], b[3]]))
    }

    /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        (
            Self::from_repr_unchecked(token, T::from_array(token, [a[0], b[0], a[1], b[1]])),
            Self::from_repr_unchecked(token, T::from_array(token, [a[2], b[2], a[3], b[3]])),
        )
    }

    // ====== 4-Channel Interleave/Deinterleave ======

    /// Deinterleave 4 RGBA pixels from AoS to SoA format.
    ///
    /// Input: 4 vectors, each containing one pixel `[R, G, B, A]`.
    /// Output: 4 vectors, each containing one channel across all pixels.
    ///
    /// This is equivalent to `transpose_4x4_copy`.
    #[inline(always)]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(rgba)
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: 4 vectors, each containing one channel across pixels.
    /// Output: 4 vectors, each containing one complete RGBA pixel.
    ///
    /// This is the inverse of `deinterleave_4ch` (also equivalent to transpose).
    #[inline(always)]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(channels)
    }

    // ====== RGBA Load/Store ======

    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (token-gated).
    ///
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_rgba_u8_t(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        let r: [f32; 4] = core::array::from_fn(|i| rgba[i * 4] as f32);
        let g: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
        let b: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
        let a: [f32; 4] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
        (
            Self::from_repr_unchecked(token, T::from_array(token, r)),
            Self::from_repr_unchecked(token, T::from_array(token, g)),
            Self::from_repr_unchecked(token, T::from_array(token, b)),
            Self::from_repr_unchecked(token, T::from_array(token, a)),
        )
    }

    /// Load 4 packed RGBA byte pixels. See [`Self::load_rgba_u8_t`].
    #[deprecated(since = "0.9.32", note = "Use load_rgba_u8_t(token, rgba).")]
    #[inline(always)]
    pub fn load_4_rgba_u8_t(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_rgba_u8_t(token, rgba)
    }

    /// Store 4 packed RGBA byte pixels from planar channels.
    ///
    /// Rounds ties to even, saturates to `0..=255`, and maps NaN to zero.
    #[inline(always)]
    pub fn store_rgba_u8(r: Self, g: Self, b: Self, a: Self, out: &mut [u8; 16]) {
        T::store_rgba_u8_into(r.1, r.0, g.0, b.0, a.0, out);
    }

    /// Store 4 packed RGBA byte pixels. See [`Self::store_rgba_u8`].
    #[deprecated(since = "0.9.32", note = "Use store_rgba_u8(r, g, b, a, out).")]
    #[inline(always)]
    pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {
        let mut out = [0u8; 16];
        Self::store_rgba_u8(r, g, b, a, &mut out);
        out
    }

    // ====== Matrix Transpose ======

    /// Transpose a 4x4 matrix represented as 4 row vectors (in-place).
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline(always)]
    pub fn transpose_4x4(rows: &mut [Self; 4]) {
        *rows = Self::transpose_4x4_copy(*rows);
    }

    /// Transpose a 4x4 matrix, returning the transposed rows.
    #[inline(always)]
    pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {
        let token = rows[0].1;
        T::transpose_4x4_repr(token, rows.map(|row| row.0))
            .map(|row| Self::from_repr_unchecked(token, row))
    }
}

// ============================================================================
// Cross-type bitcast references (require F32x4Convert)
// ============================================================================

impl<T: crate::simd::backends::F32x4Convert> f32x4<T> {
    /// Reinterpret bits as `&i32x4<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32(&self) -> &super::i32x4<T> {
        crate::simd_storage::vector_view(self.1, &self.0)
    }

    /// Reinterpret bits as `&mut i32x4<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x4<T> {
        crate::simd_storage::vector_view_mut(self.1, &mut self.0)
    }
}
// Generated deprecated token-constructor forwarders. Do not edit.
impl<T: F32x4Backend> f32x4<T> {
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_bytes_t`].\n\nUse `from_bytes_t` to keep explicit-token construction when `from_bytes` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_bytes_t(token, bytes); from_bytes becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_bytes(token: T, bytes: &[u8; 16]) -> Self {
        Self::from_bytes_t(token, bytes)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_bytes_owned_t`].\n\nUse `from_bytes_owned_t` to keep explicit-token construction when `from_bytes_owned` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_bytes_owned_t(token, bytes); from_bytes_owned becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_bytes_owned(token: T, bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_t(token, bytes)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::cast_slice_t`].\n\nUse `cast_slice_t` to keep explicit-token construction when `cast_slice` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use cast_slice_t(token, slice); cast_slice becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn cast_slice(token: T, slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::cast_slice_mut_t`].\n\nUse `cast_slice_mut_t` to keep explicit-token construction when `cast_slice_mut` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use cast_slice_mut_t(token, slice); cast_slice_mut becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn cast_slice_mut(token: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_t(token, slice)
    }
    #[inline(always)]
    #[doc = "Deprecated token-taking spelling of [`Self::from_u8_t`].\n\nUse `from_u8_t` to keep explicit-token construction when `from_u8` becomes tokenless in magetypes 0.10."]
    #[deprecated(
        note = "Use from_u8_t(token, bytes); from_u8 becomes tokenless in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn from_u8(token: T, bytes: &[u8; 4]) -> Self {
        Self::from_u8_t(token, bytes)
    }
    #[inline(always)]
    #[allow(deprecated)]
    #[doc = "Deprecated spelling of [`Self::load_rgba_u8_t`]."]
    #[deprecated(
        note = "Use load_rgba_u8_t(token, rgba); load_4_rgba_u8 will be removed in magetypes 0.10."
    )]
    #[forbid(unsafe_code)]
    pub fn load_4_rgba_u8(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_t(token, rgba)
    }
}

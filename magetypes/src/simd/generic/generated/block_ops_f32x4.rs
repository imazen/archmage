//! Block, view, and image operations for `f32x4<T>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x4Backend;
use crate::simd::generic::f32x4;

impl<T: F32x4Backend> f32x4<T> {
    // ====== Array/Byte Views ======

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        // SAFETY: f32x4<T> is repr(transparent) over T::Repr, layout-compatible with [f32; 4]
        unsafe { &*core::ptr::from_ref(self).cast::<[f32; 4]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[f32; 4]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: f32x4<T> is exactly 16 bytes for all backends
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 16]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: f32x4<T> is exactly 16 bytes for all backends
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 16]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 16]) -> Self {
        // SAFETY: f32x4<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 16]) -> Self {
        // SAFETY: f32x4<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[f32]) -> Option<&[Self]> {
        if !slice.len().is_multiple_of(4) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        if !slice.len().is_multiple_of(4) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }

    // ====== u8 Conversions ======

    /// Load 4 u8 values and convert to f32x4.
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
    }

    /// Convert to 4 u8 values with saturation.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 4] {
        let arr = self.to_array();
        core::array::from_fn(|i| arr[i].round().clamp(0.0, 255.0) as u8)
    }

    // ====== Interleave Operations ======

    /// Interleave low elements.
    ///
    /// ```text
    /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a0,b0,a1,b1]
    /// ```
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]]))
    }

    /// Interleave high elements.
    ///
    /// ```text
    /// [a0,a1,a2,a3] + [b0,b1,b2,b3] → [a2,b2,a3,b3]
    /// ```
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]]))
    }

    /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let a = self.to_array();
        let b = other.to_array();
        (
            Self::from_repr_unchecked(T::from_array([a[0], b[0], a[1], b[1]])),
            Self::from_repr_unchecked(T::from_array([a[2], b[2], a[3], b[3]])),
        )
    }

    // ====== 4-Channel Interleave/Deinterleave ======

    /// Deinterleave 4 RGBA pixels from AoS to SoA format.
    ///
    /// Input: 4 vectors, each containing one pixel `[R, G, B, A]`.
    /// Output: 4 vectors, each containing one channel across all pixels.
    ///
    /// This is equivalent to `transpose_4x4_copy`.
    #[inline]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(rgba)
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: 4 vectors, each containing one channel across pixels.
    /// Output: 4 vectors, each containing one complete RGBA pixel.
    ///
    /// This is the inverse of `deinterleave_4ch` (also equivalent to transpose).
    #[inline]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        Self::transpose_4x4_copy(channels)
    }

    // ====== RGBA Load/Store ======

    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors.
    ///
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
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
    }

    /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
    #[inline]
    pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {
        let rv = r.to_array();
        let gv = g.to_array();
        let bv = b.to_array();
        let av = a.to_array();
        let mut out = [0u8; 16];
        for i in 0..4 {
            out[i * 4] = rv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 1] = gv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 2] = bv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 3] = av[i].round().clamp(0.0, 255.0) as u8;
        }
        out
    }

    // ====== Matrix Transpose ======

    /// Transpose a 4x4 matrix represented as 4 row vectors (in-place).
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline]
    pub fn transpose_4x4(rows: &mut [Self; 4]) {
        let a = rows[0].to_array();
        let b = rows[1].to_array();
        let c = rows[2].to_array();
        let d = rows[3].to_array();
        rows[0] = Self::from_repr_unchecked(T::from_array([a[0], b[0], c[0], d[0]]));
        rows[1] = Self::from_repr_unchecked(T::from_array([a[1], b[1], c[1], d[1]]));
        rows[2] = Self::from_repr_unchecked(T::from_array([a[2], b[2], c[2], d[2]]));
        rows[3] = Self::from_repr_unchecked(T::from_array([a[3], b[3], c[3], d[3]]));
    }

    /// Transpose a 4x4 matrix, returning the transposed rows.
    #[inline]
    pub fn transpose_4x4_copy(rows: [Self; 4]) -> [Self; 4] {
        let mut result = rows;
        Self::transpose_4x4(&mut result);
        result
    }
}

// ============================================================================
// Cross-type bitcast references (require F32x4Convert)
// ============================================================================

impl<T: crate::simd::backends::F32x4Convert> f32x4<T> {
    /// Reinterpret bits as `&i32x4<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32(&self) -> &super::i32x4<T> {
        // SAFETY: f32x4<T> and i32x4<T> are both repr(transparent) with same size
        unsafe { &*core::ptr::from_ref(self).cast::<super::i32x4<T>>() }
    }

    /// Reinterpret bits as `&mut i32x4<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x4<T> {
        // SAFETY: f32x4<T> and i32x4<T> are both repr(transparent) with same size
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::i32x4<T>>() }
    }
}

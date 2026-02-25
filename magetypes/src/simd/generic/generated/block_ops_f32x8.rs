//! Block, view, and image operations for `f32x8<T>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x8Backend;
use crate::simd::generic::f32x8;

impl<T: F32x8Backend> f32x8<T> {
    // ====== Array/Byte Views ======

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 8] {
        // SAFETY: f32x8<T> is repr(transparent) over T::Repr, layout-compatible with [f32; 8]
        unsafe { &*core::ptr::from_ref(self).cast::<[f32; 8]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 8] {
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[f32; 8]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        // SAFETY: f32x8<T> is exactly 32 bytes for all backends
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 32]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        // SAFETY: f32x8<T> is exactly 32 bytes for all backends
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 32]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 32]) -> Self {
        // SAFETY: f32x8<T> is exactly 32 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 32]) -> Self {
        // SAFETY: f32x8<T> is exactly 32 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[f32]) -> Option<&[Self]> {
        if !slice.len().is_multiple_of(8) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        if !slice.len().is_multiple_of(8) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }

    // ====== u8 Conversions ======

    /// Load 8 u8 values and convert to f32x8.
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 8]) -> Self {
        Self::from_repr_unchecked(T::from_array(core::array::from_fn(|i| bytes[i] as f32)))
    }

    /// Convert to 8 u8 values with saturation.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 8] {
        let arr = self.to_array();
        core::array::from_fn(|i| arr[i].round().clamp(0.0, 255.0) as u8)
    }

    // ====== Interleave Operations ======

    /// Interleave low elements within 128-bit lanes.
    ///
    /// ```text
    /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
    /// → [a0,b0,a1,b1,a4,b4,a5,b5]
    /// ```
    #[inline(always)]
    pub fn interleave_lo(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(T::from_array([
            a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5],
        ]))
    }

    /// Interleave high elements within 128-bit lanes.
    ///
    /// ```text
    /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
    /// → [a2,b2,a3,b3,a6,b6,a7,b7]
    /// ```
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(T::from_array([
            a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7],
        ]))
    }

    /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
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
    }

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
    #[inline]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
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
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: `[R, G, B, A]` — one f32x8 per channel.
    /// Output: 4 f32x8 vectors in interleaved AoS format.
    ///
    /// This is the inverse of `deinterleave_4ch`.
    #[inline]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        let r = channels[0].to_array();
        let g = channels[1].to_array();
        let b = channels[2].to_array();
        let a = channels[3].to_array();
        core::array::from_fn(|i| {
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
        })
    }

    // ====== RGBA Load/Store ======

    /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
    ///
    /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x8 with values in `[0.0, 255.0]`.
    #[inline]
    pub fn load_8_rgba_u8(rgba: &[u8; 32]) -> (Self, Self, Self, Self) {
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
    }

    /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
    #[inline]
    pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {
        let rv = r.to_array();
        let gv = g.to_array();
        let bv = b.to_array();
        let av = a.to_array();
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[i * 4] = rv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 1] = gv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 2] = bv[i].round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 3] = av[i].round().clamp(0.0, 255.0) as u8;
        }
        out
    }

    // ====== Matrix Transpose ======

    /// Transpose an 8x8 matrix represented as 8 row vectors (in-place).
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline]
    pub fn transpose_8x8(rows: &mut [Self; 8]) {
        let r: [[f32; 8]; 8] = core::array::from_fn(|i| rows[i].to_array());
        for i in 0..8 {
            rows[i] = Self::from_repr_unchecked(T::from_array(core::array::from_fn(|j| r[j][i])));
        }
    }

    /// Transpose an 8x8 matrix, returning the transposed rows.
    #[inline]
    pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {
        let mut result = rows;
        Self::transpose_8x8(&mut result);
        result
    }

    /// Load an 8x8 f32 block from a contiguous array into 8 row vectors.
    #[inline]
    pub fn load_8x8(block: &[f32; 64]) -> [Self; 8] {
        core::array::from_fn(|i| {
            let arr: [f32; 8] = block[i * 8..][..8].try_into().unwrap();
            Self::from_repr_unchecked(T::from_array(arr))
        })
    }

    /// Store 8 row vectors to a contiguous 8x8 f32 block.
    #[inline]
    pub fn store_8x8(rows: &[Self; 8], block: &mut [f32; 64]) {
        for i in 0..8 {
            let arr = rows[i].to_array();
            block[i * 8..][..8].copy_from_slice(&arr);
        }
    }
}

// ============================================================================
// Cross-type bitcast references (require F32x8Convert)
// ============================================================================

impl<T: crate::simd::backends::F32x8Convert> f32x8<T> {
    /// Reinterpret bits as `&i32x8<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32(&self) -> &super::i32x8<T> {
        // SAFETY: f32x8<T> and i32x8<T> are both repr(transparent) with same size
        unsafe { &*core::ptr::from_ref(self).cast::<super::i32x8<T>>() }
    }

    /// Reinterpret bits as `&mut i32x8<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x8<T> {
        // SAFETY: f32x8<T> and i32x8<T> are both repr(transparent) with same size
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::i32x8<T>>() }
    }
}

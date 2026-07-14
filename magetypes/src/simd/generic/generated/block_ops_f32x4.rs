//! Block, view, and image operations for `f32x4<T>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x4Backend;
use crate::simd::generic::f32x4;

impl<T: F32x4Backend> f32x4<T> {
    // ====== Array/Byte Views ======
    //
    // Layout note for every view below: f32x4<T> is #[repr(C)]
    // (T::Repr, token) where the token is a 1-ZST, so its size and
    // layout are exactly T::Repr's. Each method opens with an
    // inline-const assert tying size_of::<Self>() to the literal
    // array size it casts to — evaluated per backend at
    // monomorphization, so a mis-sized future Repr is a compile
    // error, never an out-of-bounds view.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 4]>()) };
        // SAFETY: size asserted above; #[repr(C)] over T::Repr (bag
        // of f32 lanes) + trailing ZST token, so element layout
        // matches [f32; 4].
        unsafe { &*core::ptr::from_ref(self).cast::<[f32; 4]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 4]>()) };
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[f32; 4]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        const { assert!(core::mem::size_of::<Self>() == 16) };
        // SAFETY: size asserted above; every byte of a SIMD vector
        // repr is initialized POD.
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 16]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        const { assert!(core::mem::size_of::<Self>() == 16) };
        // SAFETY: size asserted above; all bit patterns are valid
        // for f32 lanes, so writes through the view stay sound.
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 16]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 16]) -> Self {
        const { assert!(core::mem::size_of::<Self>() == 16) };
        // SAFETY: sizes match (asserted above); all bit patterns are
        // valid f32 lanes and the token is a ZST; transmute_copy
        // reads unaligned.
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 16]) -> Self {
        const { assert!(core::mem::size_of::<Self>() == 16) };
        // SAFETY: as from_bytes — sizes asserted, all bit patterns
        // valid, unaligned read.
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[f32]) -> Option<&[Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 4]>()) };
        if !slice.len().is_multiple_of(4) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime, so the reinterpreted slice covers
        // exactly the same bytes.
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 4]>()) };
        if !slice.len().is_multiple_of(4) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 4;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime; exclusive borrow carries over.
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }

    // ====== u8 Conversions ======

    /// Load 4 u8 values and convert to f32x4 (token-gated).
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8(token: T, bytes: &[u8; 4]) -> Self {
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
    pub fn load_4_rgba_u8(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
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

    /// Interleave 4 f32x4 channels and store as 4 RGBA u8 pixels.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    /// Output: 16 bytes = 4 RGBA pixels in interleaved format.
    #[inline(always)]
    pub fn store_4_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 16] {
        T::store_rgba_bytes(r.1, r.0, g.0, b.0, a.0)
    }

    // ====== Matrix Transpose ======

    /// Transpose a 4x4 matrix represented as 4 row vectors (in-place).
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline(always)]
    pub fn transpose_4x4(rows: &mut [Self; 4]) {
        let token = rows[0].1;
        let a = rows[0].to_array();
        let b = rows[1].to_array();
        let c = rows[2].to_array();
        let d = rows[3].to_array();
        rows[0] = Self::from_repr_unchecked(token, T::from_array(token, [a[0], b[0], c[0], d[0]]));
        rows[1] = Self::from_repr_unchecked(token, T::from_array(token, [a[1], b[1], c[1], d[1]]));
        rows[2] = Self::from_repr_unchecked(token, T::from_array(token, [a[2], b[2], c[2], d[2]]));
        rows[3] = Self::from_repr_unchecked(token, T::from_array(token, [a[3], b[3], c[3], d[3]]));
    }

    /// Transpose a 4x4 matrix, returning the transposed rows.
    #[inline(always)]
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
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::i32x4<T>>()) };
        // SAFETY: sizes asserted equal above; both are #[repr(C)]
        // (Repr, token-ZST) wrappers of same-width lane vectors, and
        // integer lanes accept all bit patterns.
        unsafe { &*core::ptr::from_ref(self).cast::<super::i32x4<T>>() }
    }

    /// Reinterpret bits as `&mut i32x4<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x4<T> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::i32x4<T>>()) };
        // SAFETY: as bitcast_ref_i32; float lanes likewise accept
        // all bit patterns written back through the view.
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::i32x4<T>>() }
    }
}

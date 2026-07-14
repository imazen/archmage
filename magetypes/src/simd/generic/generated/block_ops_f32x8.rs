//! Block, view, and image operations for `f32x8<T>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x8Backend;
use crate::simd::generic::f32x8;

impl<T: F32x8Backend> f32x8<T> {
    // ====== Array/Byte Views ======
    //
    // Layout note for every view below: f32x8<T> is #[repr(C)]
    // (T::Repr, token) where the token is a 1-ZST, so its size and
    // layout are exactly T::Repr's. Each method opens with an
    // inline-const assert tying size_of::<Self>() to the literal
    // array size it casts to — evaluated per backend at
    // monomorphization, so a mis-sized future Repr is a compile
    // error, never an out-of-bounds view.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 8] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 8]>()) };
        // SAFETY: size asserted above; #[repr(C)] over T::Repr (bag
        // of f32 lanes) + trailing ZST token, so element layout
        // matches [f32; 8].
        unsafe { &*core::ptr::from_ref(self).cast::<[f32; 8]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 8] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 8]>()) };
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[f32; 8]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        const { assert!(core::mem::size_of::<Self>() == 32) };
        // SAFETY: size asserted above; every byte of a SIMD vector
        // repr is initialized POD.
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 32]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        const { assert!(core::mem::size_of::<Self>() == 32) };
        // SAFETY: size asserted above; all bit patterns are valid
        // for f32 lanes, so writes through the view stay sound.
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 32]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 32]) -> Self {
        const { assert!(core::mem::size_of::<Self>() == 32) };
        // SAFETY: sizes match (asserted above); all bit patterns are
        // valid f32 lanes and the token is a ZST; transmute_copy
        // reads unaligned.
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 32]) -> Self {
        const { assert!(core::mem::size_of::<Self>() == 32) };
        // SAFETY: as from_bytes — sizes asserted, all bit patterns
        // valid, unaligned read.
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[f32]) -> Option<&[Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 8]>()) };
        if !slice.len().is_multiple_of(8) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime, so the reinterpreted slice covers
        // exactly the same bytes.
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[f32; 8]>()) };
        if !slice.len().is_multiple_of(8) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 8;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime; exclusive borrow carries over.
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }

    // ====== u8 Conversions ======

    /// Load 8 u8 values and convert to f32x8 (token-gated).
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8(token: T, bytes: &[u8; 8]) -> Self {
        Self::from_repr_unchecked(
            token,
            T::from_array(token, core::array::from_fn(|i| bytes[i] as f32)),
        )
    }

    /// Convert to 8 u8 values with saturation.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    #[inline(always)]
    pub fn to_u8(self) -> [u8; 8] {
        T::to_u8_bytes(self.1, self.0)
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
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(
            token,
            T::from_array(token, [a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]]),
        )
    }

    /// Interleave high elements within 128-bit lanes.
    ///
    /// ```text
    /// [a0,a1,a2,a3,a4,a5,a6,a7] + [b0,b1,b2,b3,b4,b5,b6,b7]
    /// → [a2,b2,a3,b3,a6,b6,a7,b7]
    /// ```
    #[inline(always)]
    pub fn interleave_hi(self, other: Self) -> Self {
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        Self::from_repr_unchecked(
            token,
            T::from_array(token, [a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]]),
        )
    }

    /// Interleave two vectors: returns `(interleave_lo, interleave_hi)`.
    #[inline(always)]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        let token = self.1;
        let a = self.to_array();
        let b = other.to_array();
        (
            Self::from_repr_unchecked(
                token,
                T::from_array(token, [a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5]]),
            ),
            Self::from_repr_unchecked(
                token,
                T::from_array(token, [a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7]]),
            ),
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
    #[inline(always)]
    pub fn deinterleave_4ch(rgba: [Self; 4]) -> [Self; 4] {
        let token = rgba[0].1;
        let v: [[f32; 8]; 4] = core::array::from_fn(|i| rgba[i].to_array());
        // Each input vector has 2 RGBA pixels (4 elements each)
        // Pixel i: v[i/2][(i%2)*4 + channel]
        let r: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4]);
        let g: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 1]);
        let b: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 2]);
        let a: [f32; 8] = core::array::from_fn(|i| v[i / 2][(i % 2) * 4 + 3]);
        [
            Self::from_repr_unchecked(token, T::from_array(token, r)),
            Self::from_repr_unchecked(token, T::from_array(token, g)),
            Self::from_repr_unchecked(token, T::from_array(token, b)),
            Self::from_repr_unchecked(token, T::from_array(token, a)),
        ]
    }

    /// Interleave 4 channels from SoA to AoS format.
    ///
    /// Input: `[R, G, B, A]` — one f32x8 per channel.
    /// Output: 4 f32x8 vectors in interleaved AoS format.
    ///
    /// This is the inverse of `deinterleave_4ch`.
    #[inline(always)]
    pub fn interleave_4ch(channels: [Self; 4]) -> [Self; 4] {
        let token = channels[0].1;
        let r = channels[0].to_array();
        let g = channels[1].to_array();
        let b = channels[2].to_array();
        let a = channels[3].to_array();
        core::array::from_fn(|i| {
            let base = i * 2;
            Self::from_repr_unchecked(
                token,
                T::from_array(
                    token,
                    [
                        r[base],
                        g[base],
                        b[base],
                        a[base],
                        r[base + 1],
                        g[base + 1],
                        b[base + 1],
                        a[base + 1],
                    ],
                ),
            )
        })
    }

    // ====== RGBA Load/Store ======

    /// Load 8 RGBA u8 pixels and deinterleave to 4 f32x8 channel vectors.
    ///
    /// Input: 32 bytes = 8 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x8 with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_8_rgba_u8(token: T, rgba: &[u8; 32]) -> (Self, Self, Self, Self) {
        let r: [f32; 8] = core::array::from_fn(|i| rgba[i * 4] as f32);
        let g: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 1] as f32);
        let b: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 2] as f32);
        let a: [f32; 8] = core::array::from_fn(|i| rgba[i * 4 + 3] as f32);
        (
            Self::from_repr_unchecked(token, T::from_array(token, r)),
            Self::from_repr_unchecked(token, T::from_array(token, g)),
            Self::from_repr_unchecked(token, T::from_array(token, b)),
            Self::from_repr_unchecked(token, T::from_array(token, a)),
        )
    }

    /// Interleave 4 f32x8 channels and store as 8 RGBA u8 pixels.
    ///
    /// Values are rounded and clamped to `[0, 255]`.
    /// Output: 32 bytes = 8 RGBA pixels in interleaved format.
    #[inline(always)]
    pub fn store_8_rgba_u8(r: Self, g: Self, b: Self, a: Self) -> [u8; 32] {
        T::store_rgba_bytes(r.1, r.0, g.0, b.0, a.0)
    }

    // ====== Matrix Transpose ======

    /// Transpose an 8x8 matrix represented as 8 row vectors (in-place).
    ///
    /// After transpose, `rows[i][j]` becomes `rows[j][i]`.
    #[inline(always)]
    pub fn transpose_8x8(rows: &mut [Self; 8]) {
        let token = rows[0].1;
        let reprs = T::transpose_8x8_repr(token, core::array::from_fn(|i| rows[i].0));
        for i in 0..8 {
            rows[i] = Self::from_repr_unchecked(token, reprs[i]);
        }
    }

    /// Transpose an 8x8 matrix, returning the transposed rows.
    #[inline(always)]
    pub fn transpose_8x8_copy(rows: [Self; 8]) -> [Self; 8] {
        let mut result = rows;
        Self::transpose_8x8(&mut result);
        result
    }

    /// Load an 8x8 f32 block from a contiguous array into 8 row vectors.
    #[inline(always)]
    pub fn load_8x8(token: T, block: &[f32; 64]) -> [Self; 8] {
        core::array::from_fn(|i| {
            let arr: [f32; 8] = block[i * 8..][..8].try_into().unwrap();
            Self::from_repr_unchecked(token, T::from_array(token, arr))
        })
    }

    /// Store 8 row vectors to a contiguous 8x8 f32 block.
    #[inline(always)]
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
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::i32x8<T>>()) };
        // SAFETY: sizes asserted equal above; both are #[repr(C)]
        // (Repr, token-ZST) wrappers of same-width lane vectors, and
        // integer lanes accept all bit patterns.
        unsafe { &*core::ptr::from_ref(self).cast::<super::i32x8<T>>() }
    }

    /// Reinterpret bits as `&mut i32x8<T>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x8<T> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::i32x8<T>>()) };
        // SAFETY: as bitcast_ref_i32; float lanes likewise accept
        // all bit patterns written back through the view.
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::i32x8<T>>() }
    }
}

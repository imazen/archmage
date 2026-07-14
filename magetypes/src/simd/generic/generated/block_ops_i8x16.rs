//! Block and view operations for `i8x16<T>`.
//!
//! Array/byte views and slice casting.

use crate::simd::backends::I8x16Backend;
use crate::simd::generic::i8x16;

impl<T: I8x16Backend> i8x16<T> {
    // ====== Array/Byte Views ======
    //
    // Layout note for every view below: i8x16<T> is #[repr(C)]
    // (T::Repr, token) where the token is a 1-ZST, so its size and
    // layout are exactly T::Repr's. Each method opens with an
    // inline-const assert tying size_of::<Self>() to the literal
    // array size it casts to — evaluated per backend at
    // monomorphization, so a mis-sized future Repr is a compile
    // error, never an out-of-bounds view.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[i8; 16] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[i8; 16]>()) };
        // SAFETY: size asserted above; #[repr(C)] over T::Repr (bag
        // of i8 lanes) + trailing ZST token, so element layout
        // matches [i8; 16].
        unsafe { &*core::ptr::from_ref(self).cast::<[i8; 16]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i8; 16] {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[i8; 16]>()) };
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[i8; 16]>() }
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
        // for i8 lanes, so writes through the view stay sound.
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 16]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 16]) -> Self {
        const { assert!(core::mem::size_of::<Self>() == 16) };
        // SAFETY: sizes match (asserted above); all bit patterns are
        // valid i8 lanes and the token is a ZST; transmute_copy
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
    /// Returns `None` if length is not a multiple of 16 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[i8]) -> Option<&[Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[i8; 16]>()) };
        if !slice.len().is_multiple_of(16) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime, so the reinterpreted slice covers
        // exactly the same bytes.
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 16 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [i8]) -> Option<&mut [Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[i8; 16]>()) };
        if !slice.len().is_multiple_of(16) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 16;
        // SAFETY: element size asserted above, alignment and length
        // checked at runtime; exclusive borrow carries over.
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }
}

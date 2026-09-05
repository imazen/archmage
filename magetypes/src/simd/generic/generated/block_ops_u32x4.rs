//! Block and view operations for `u32x4<T>`.
//!
//! Array/byte views, slice casting, and cross-type bitcast to f32x4.

use crate::simd::backends::U32x4Backend;
use crate::simd::generic::u32x4;

impl<T: U32x4Backend> u32x4<T> {
    // ====== Array/Byte Views ======
    //
    // Views borrow only raw storage. Checked helpers enforce size and
    // alignment without exposing or manufacturing the token.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 4] {
        crate::simd_storage::view(&self.0)
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u32; 4] {
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
    pub fn from_bytes(token: T, bytes: &[u8; 16]) -> Self {
        Self(crate::simd_storage::copy(bytes), token)
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(token: T, bytes: [u8; 16]) -> Self {
        Self(crate::simd_storage::cast(bytes), token)
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[u32]) -> Option<&[Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[u32; 4]>()) };
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
    pub fn cast_slice_mut(_: T, slice: &mut [u32]) -> Option<&mut [Self]> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<[u32; 4]>()) };
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
}

// ============================================================================
// Cross-type bitcast to f32x4 (requires F32x4Backend)
// ============================================================================

impl<T: U32x4Backend + crate::simd::backends::F32x4Backend> u32x4<T> {
    /// Bitcast to f32x4 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> super::f32x4<T> {
        super::f32x4(crate::simd_storage::cast(self.0), self.1)
    }

    /// Bitcast to f32x4 by reference (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x4(&self) -> &super::f32x4<T> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::f32x4<T>>()) };
        // SAFETY: sizes asserted equal above; both are #[repr(C)]
        // (Repr, token-ZST) wrappers and f32 lanes accept all bit
        // patterns.
        unsafe { &*core::ptr::from_ref(self).cast::<super::f32x4<T>>() }
    }

    /// Bitcast to f32x4 by mutable reference (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x4(&mut self) -> &mut super::f32x4<T> {
        const { assert!(core::mem::size_of::<Self>() == core::mem::size_of::<super::f32x4<T>>()) };
        // SAFETY: as bitcast_ref_f32x4; u32 lanes likewise accept
        // all bit patterns written back.
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::f32x4<T>>() }
    }
}

//! Block and view operations for `u32x4<T>`.
//!
//! Array/byte views, slice casting, and cross-type bitcast to f32x4.

use crate::simd::backends::U32x4Backend;
use crate::simd::generic::u32x4;

impl<T: U32x4Backend> u32x4<T> {
    // ====== Array/Byte Views ======

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[u32; 4] {
        // SAFETY: u32x4<T> is repr(transparent) over T::Repr, layout-compatible with [u32; 4]
        unsafe { &*core::ptr::from_ref(self).cast::<[u32; 4]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [u32; 4] {
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u32; 4]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: u32x4<T> is exactly 16 bytes for all backends
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 16]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: u32x4<T> is exactly 16 bytes for all backends
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 16]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 16]) -> Self {
        // SAFETY: u32x4<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 16]) -> Self {
        // SAFETY: u32x4<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[u32]) -> Option<&[Self]> {
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
    pub fn cast_slice_mut(_: T, slice: &mut [u32]) -> Option<&mut [Self]> {
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
}

// ============================================================================
// Cross-type bitcast to f32x4 (requires F32x4Backend)
// ============================================================================

impl<T: U32x4Backend + crate::simd::backends::F32x4Backend> u32x4<T> {
    /// Bitcast to f32x4 (reinterpret bits, no conversion).
    #[inline(always)]
    pub fn bitcast_f32x4(self) -> super::f32x4<T> {
        // SAFETY: u32x4<T> and f32x4<T> are both exactly 16 bytes
        unsafe { core::mem::transmute_copy(&self) }
    }

    /// Bitcast to f32x4 by reference (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_f32x4(&self) -> &super::f32x4<T> {
        // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
        unsafe { &*core::ptr::from_ref(self).cast::<super::f32x4<T>>() }
    }

    /// Bitcast to f32x4 by mutable reference (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_f32x4(&mut self) -> &mut super::f32x4<T> {
        // SAFETY: u32x4 and f32x4 are both repr(transparent) with same size (16 bytes)
        unsafe { &mut *core::ptr::from_mut(self).cast::<super::f32x4<T>>() }
    }
}

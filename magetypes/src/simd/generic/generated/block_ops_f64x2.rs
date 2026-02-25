//! Block and view operations for `f64x2<T>`.
//!
//! Array/byte views and slice casting.

use crate::simd::backends::F64x2Backend;
use crate::simd::generic::f64x2;

impl<T: F64x2Backend> f64x2<T> {
    // ====== Array/Byte Views ======

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f64; 2] {
        // SAFETY: f64x2<T> is repr(transparent) over T::Repr, layout-compatible with [f64; 2]
        unsafe { &*core::ptr::from_ref(self).cast::<[f64; 2]>() }
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f64; 2] {
        // SAFETY: same layout guarantee as as_array
        unsafe { &mut *core::ptr::from_mut(self).cast::<[f64; 2]>() }
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: f64x2<T> is exactly 16 bytes for all backends
        unsafe { &*core::ptr::from_ref(self).cast::<[u8; 16]>() }
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: f64x2<T> is exactly 16 bytes for all backends
        unsafe { &mut *core::ptr::from_mut(self).cast::<[u8; 16]>() }
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(_: T, bytes: &[u8; 16]) -> Self {
        // SAFETY: f64x2<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(bytes) }
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(_: T, bytes: [u8; 16]) -> Self {
        // SAFETY: f64x2<T> is exactly 16 bytes; transmute_copy uses read_unaligned
        unsafe { core::mem::transmute_copy(&bytes) }
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 2 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(_: T, slice: &[f64]) -> Option<&[Self]> {
        if !slice.len().is_multiple_of(2) {
            return None;
        }
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts(ptr.cast::<Self>(), len) })
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 2 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(_: T, slice: &mut [f64]) -> Option<&mut [Self]> {
        if !slice.len().is_multiple_of(2) {
            return None;
        }
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {
            return None;
        }
        let len = slice.len() / 2;
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Self>(), len) })
    }
}

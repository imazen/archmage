//! Block and view operations for `i32x8<T>`.
//!
//! Array/byte views and slice casting.

use crate::simd::backends::I32x8Backend;
use crate::simd::generic::i32x8;

impl<T: I32x8Backend> i32x8<T> {
    // ====== Array/Byte Views ======
    //
    // Views borrow only raw storage. Checked helpers enforce size and
    // alignment without exposing or manufacturing the token.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 8] {
        crate::simd_storage::view(&self.0)
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i32; 8] {
        crate::simd_storage::view_mut(&mut self.0)
    }

    /// View as byte array.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; 32] {
        crate::simd_storage::view(&self.0)
    }

    /// View as mutable byte array.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        crate::simd_storage::view_mut(&mut self.0)
    }

    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(token: T, bytes: &[u8; 32]) -> Self {
        Self(crate::simd_storage::copy(bytes), token)
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(token: T, bytes: [u8; 32]) -> Self {
        Self(crate::simd_storage::cast(bytes), token)
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(token: T, slice: &[i32]) -> Option<&[Self]> {
        crate::simd_storage::vector_slice::<_, Self, 8>(token, slice)
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 8 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(token: T, slice: &mut [i32]) -> Option<&mut [Self]> {
        crate::simd_storage::vector_slice_mut::<_, Self, 8>(token, slice)
    }
}
// Generated explicit-token migration aliases. Do not edit.
impl<T: I32x8Backend> i32x8<T> {
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::from_bytes`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn from_bytes_t(token: T, bytes: &[u8; 32]) -> Self {
        Self::from_bytes(token, bytes)
    }
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::from_bytes_owned`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn from_bytes_owned_t(token: T, bytes: [u8; 32]) -> Self {
        Self::from_bytes_owned(token, bytes)
    }
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::cast_slice`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn cast_slice_t(token: T, slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice(token, slice)
    }
    #[inline(always)]
    #[doc = "Explicit-token alias of [`Self::cast_slice_mut`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."]
    #[forbid(unsafe_code)]
    pub fn cast_slice_mut_t(token: T, slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut(token, slice)
    }
}

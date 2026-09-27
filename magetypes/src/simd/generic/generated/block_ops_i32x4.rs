//! Block and view operations for `i32x4<T, M>`.
//!
//! Array/byte views and slice casting.

use crate::simd::backends::I32x4Backend;
use crate::simd::generic::core_types::i32x4;

impl<M: crate::simd::generic::ConstructorMode, T: I32x4Backend> i32x4<T, M> {
    // ====== Array/Byte Views ======
    //
    // Views borrow only raw storage. Checked helpers enforce size and
    // alignment without exposing or manufacturing the token.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[i32; 4] {
        crate::simd_storage::view(&self.0)
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [i32; 4] {
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
    pub(crate) fn from_bytes_with_token(token: T, bytes: &[u8; 16]) -> Self {
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }

    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub(crate) fn from_bytes_owned_with_token(token: T, bytes: [u8; 16]) -> Self {
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }

    // ====== Slice Casting ======

    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub(crate) fn cast_slice_with_token(token: T, slice: &[i32]) -> Option<&[Self]> {
        crate::simd_storage::vector_slice::<_, Self, 4>(token, slice)
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub(crate) fn cast_slice_mut_with_token(token: T, slice: &mut [i32]) -> Option<&mut [Self]> {
        crate::simd_storage::vector_slice_mut::<_, Self, 4>(token, slice)
    }
}

#[cfg(target_arch = "aarch64")]
impl i32x4<archmage::NeonToken, crate::simd::generic::Context> {
    /// Create from byte array reference (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self::from_bytes_with_token(archmage::NeonToken::from_context(), bytes)
    }
    /// Create from owned byte array (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_with_token(archmage::NeonToken::from_context(), bytes)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn cast_slice(slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(neon)]
    pub fn cast_slice_mut(slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::NeonToken::from_context(), slice)
    }
}

#[cfg(target_arch = "wasm32")]
impl i32x4<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Create from byte array reference (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self::from_bytes_with_token(archmage::Wasm128Token::from_context(), bytes)
    }
    /// Create from owned byte array (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_with_token(archmage::Wasm128Token::from_context(), bytes)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn cast_slice(slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(wasm128)]
    pub fn cast_slice_mut(slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::Wasm128Token::from_context(), slice)
    }
}

#[cfg(target_arch = "x86_64")]
impl i32x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from byte array reference (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self::from_bytes_with_token(archmage::X64V3Token::from_context(), bytes)
    }
    /// Create from owned byte array (token-gated).
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_with_token(archmage::X64V3Token::from_context(), bytes)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn cast_slice(slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[archmage::rite(v3)]
    pub fn cast_slice_mut(slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::X64V3Token::from_context(), slice)
    }
}

impl<T: I32x4Backend> i32x4<T, crate::simd::generic::Explicit> {
    /// Create from byte array reference (token-gated).
    #[inline(always)]
    pub fn from_bytes(token: T, bytes: &[u8; 16]) -> Self {
        Self::from_bytes_with_token(token, bytes)
    }
    /// Create from owned byte array (token-gated).
    #[inline(always)]
    pub fn from_bytes_owned(token: T, bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_with_token(token, bytes)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice(token: T, slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(token, slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(token: T, slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(token, slice)
    }
}

impl i32x4<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Create from byte array reference (token-gated).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self::from_bytes_with_token(archmage::ScalarToken, bytes)
    }
    /// Create from owned byte array (token-gated).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        Self::from_bytes_owned_with_token(archmage::ScalarToken, bytes)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn cast_slice(slice: &[i32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn cast_slice_mut(slice: &mut [i32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::ScalarToken, slice)
    }
}

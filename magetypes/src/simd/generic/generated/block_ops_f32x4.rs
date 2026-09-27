//! Block, view, and image operations for `f32x4<T, M>`.
//!
//! Array/byte views, slice casting, interleave/deinterleave,
//! matrix transpose, RGBA pixel operations, and cross-type bitcast references.

use crate::simd::backends::F32x4Backend;
use crate::simd::generic::core_types::f32x4;

impl<M: crate::simd::generic::ConstructorMode, T: F32x4Backend> f32x4<T, M> {
    // ====== Array/Byte Views ======
    //
    // Views borrow only raw storage. Checked helpers enforce size and
    // alignment without exposing or manufacturing the token.

    /// Reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        crate::simd_storage::view(&self.0)
    }

    /// Mutable reference to underlying array (zero-copy).
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
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
    pub(crate) fn cast_slice_with_token(token: T, slice: &[f32]) -> Option<&[Self]> {
        crate::simd_storage::vector_slice::<_, Self, 4>(token, slice)
    }

    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    ///
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub(crate) fn cast_slice_mut_with_token(token: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        crate::simd_storage::vector_slice_mut::<_, Self, 4>(token, slice)
    }

    // ====== u8 Conversions ======

    /// Load 4 u8 values and convert to f32x4 (token-gated).
    ///
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub(crate) fn from_u8_with_token(token: T, bytes: &[u8; 4]) -> Self {
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
    pub(crate) fn load_4_rgba_u8_with_token(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
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

impl<M: crate::simd::generic::ConstructorMode, T: crate::simd::backends::F32x4Convert> f32x4<T, M> {
    /// Reinterpret bits as `&i32x4<T, M>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_ref_i32(&self) -> &super::i32x4<T, M> {
        crate::simd_storage::vector_view(self.1, &self.0)
    }

    /// Reinterpret bits as `&mut i32x4<T, M>` (zero-cost pointer cast).
    #[inline(always)]
    pub fn bitcast_mut_i32(&mut self) -> &mut super::i32x4<T, M> {
        crate::simd_storage::vector_view_mut(self.1, &mut self.0)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::Avx512Fp16Token, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::Avx512Fp16Token::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::Avx512Fp16Token::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::Avx512Fp16Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::Avx512Fp16Token::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::Avx512Fp16Token::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16"
    )]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::Avx512Fp16Token::from_context(), rgba)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::X64V4Token, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::X64V4Token::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::X64V4Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::X64V4Token::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::X64V4Token::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::X64V4Token::from_context(), rgba)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl f32x4<archmage::X64V4xToken, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::X64V4xToken::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::X64V4xToken::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::X64V4xToken::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::X64V4xToken::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::X64V4xToken::from_context(), rgba)
    }
}

#[cfg(target_arch = "aarch64")]
impl f32x4<archmage::NeonToken, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::NeonToken::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::NeonToken::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::NeonToken::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::NeonToken::from_context(), rgba)
    }
}

#[cfg(target_arch = "wasm32")]
impl f32x4<archmage::Wasm128Token, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::Wasm128Token::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::Wasm128Token::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::Wasm128Token::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::Wasm128Token::from_context(), rgba)
    }
}

#[cfg(target_arch = "x86_64")]
impl f32x4<archmage::X64V3Token, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::X64V3Token::from_context();
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::X64V3Token::from_context(), slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::X64V3Token::from_context(), bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    /// # Safety
    /// The CPU must support the enabled target features. Safe calls require a
    /// matching or stronger target-feature context, which Rust checks.
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::X64V3Token::from_context(), rgba)
    }
}

impl<T: F32x4Backend> f32x4<T, crate::simd::generic::Explicit> {
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
    pub fn cast_slice(token: T, slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(token, slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (token-gated).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[inline(always)]
    pub fn cast_slice_mut(token: T, slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(token, slice)
    }
    /// Load 4 u8 values and convert to f32x4 (token-gated).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[inline(always)]
    pub fn from_u8(token: T, bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(token, bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (token-gated).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[inline(always)]
    pub fn load_4_rgba_u8(token: T, rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(token, rgba)
    }
}

impl f32x4<archmage::ScalarToken, crate::simd::generic::Context> {
    /// Create from byte array reference (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(crate::simd_storage::copy(bytes), token)
    }
    /// Create from owned byte array (requires matching target features).
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_bytes_owned(bytes: [u8; 16]) -> Self {
        let token = archmage::ScalarToken;
        Self::new_repr(crate::simd_storage::cast(bytes), token)
    }
    /// Reinterpret a scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn cast_slice(slice: &[f32]) -> Option<&[Self]> {
        Self::cast_slice_with_token(archmage::ScalarToken, slice)
    }
    /// Reinterpret a mutable scalar slice as a SIMD vector slice (requires matching target features).
    /// Returns `None` if length is not a multiple of 4 or alignment is wrong.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn cast_slice_mut(slice: &mut [f32]) -> Option<&mut [Self]> {
        Self::cast_slice_mut_with_token(archmage::ScalarToken, slice)
    }
    /// Load 4 u8 values and convert to f32x4 (requires matching target features).
    /// Values are in `[0.0, 255.0]`. Useful for image processing.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn from_u8(bytes: &[u8; 4]) -> Self {
        Self::from_u8_with_token(archmage::ScalarToken, bytes)
    }
    /// Load 4 RGBA u8 pixels and deinterleave to 4 f32x4 channel vectors (requires matching target features).
    /// Input: 16 bytes = 4 RGBA pixels in interleaved format.
    /// Output: `(R, G, B, A)` where each is f32x4 with values in `[0.0, 255.0]`.
    #[forbid(unsafe_code)]
    #[inline(always)]
    pub fn load_4_rgba_u8(rgba: &[u8; 16]) -> (Self, Self, Self, Self) {
        Self::load_4_rgba_u8_with_token(archmage::ScalarToken, rgba)
    }
}

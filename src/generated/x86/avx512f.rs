//! Token-gated wrappers for `#[target_feature(enable = "avx512f")]` functions.
//!
//! This module contains 49 functions that are safe to call when you have a [`Avx512fToken`].
//!
//! **Auto-generated** from safe_unaligned_simd v0.2.3 - do not edit manually.
//! See `xtask/src/main.rs` for the generator.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]


#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use safe_unaligned_simd::x86::{
    Is16BitsUnaligned, Is32BitsUnaligned, Is64BitsUnaligned,
    Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned,
    Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned,
    Is128CellUnaligned, Is256CellUnaligned,
};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    Is16BitsUnaligned, Is32BitsUnaligned, Is64BitsUnaligned,
    Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned,
    Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned,
    Is128CellUnaligned, Is256CellUnaligned,
};

use crate::tokens::x86::Avx512fToken;

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi32)
#[inline(always)]
pub fn _mm512_mask_expandloadu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, src: __m512i, k: __mmask16, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask16, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi32)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, k: __mmask16, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask16, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi64)
#[inline(always)]
pub fn _mm512_mask_expandloadu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, src: __m512i, k: __mmask8, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask8, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi64)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, k: __mmask8, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_pd)
#[inline(always)]
pub fn _mm512_mask_expandloadu_pd(_token: Avx512fToken, src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_pd)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_pd(_token: Avx512fToken, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_ps)
#[inline(always)]
pub fn _mm512_mask_expandloadu_ps(_token: Avx512fToken, src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_ps)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_ps(_token: Avx512fToken, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 512-bits (composed of 16 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi32)
#[inline(always)]
pub fn _mm512_loadu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_loadu_epi32::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi32)
#[inline(always)]
pub fn _mm512_mask_loadu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, src: __m512i, k: __mmask16, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask16, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi32)
#[inline(always)]
pub fn _mm512_maskz_loadu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, k: __mmask16, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask16, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 512-bits (composed of 8 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi64)
#[inline(always)]
pub fn _mm512_loadu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_loadu_epi64::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi64)
#[inline(always)]
pub fn _mm512_mask_loadu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, src: __m512i, k: __mmask8, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask8, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi64)
#[inline(always)]
pub fn _mm512_maskz_loadu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, k: __mmask8, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Loads 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_pd)
#[inline(always)]
pub fn _mm512_loadu_pd(_token: Avx512fToken, mem_addr: &[f64; 8]) -> __m512d {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &[f64; 8]) -> __m512d {
        safe_unaligned_simd::x86_64::_mm512_loadu_pd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_pd)
#[inline(always)]
pub fn _mm512_mask_loadu_pd(_token: Avx512fToken, src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_pd)
#[inline(always)]
pub fn _mm512_maskz_loadu_pd(_token: Avx512fToken, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Loads 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_ps)
#[inline(always)]
pub fn _mm512_loadu_ps(_token: Avx512fToken, mem_addr: &[f32; 16]) -> __m512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &[f32; 16]) -> __m512 {
        safe_unaligned_simd::x86_64::_mm512_loadu_ps(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_ps)
#[inline(always)]
pub fn _mm512_mask_loadu_ps(_token: Avx512fToken, src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_ps)
#[inline(always)]
pub fn _mm512_maskz_loadu_ps(_token: Avx512fToken, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 512-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_si512)
#[inline(always)]
pub fn _mm512_loadu_si512<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_loadu_si512::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi32)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi64)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_epi64::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_pd)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_pd(_token: Avx512fToken, base_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(base_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_pd(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_ps)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_ps(_token: Avx512fToken, base_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(base_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_ps(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi32_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtepi32_storeu_epi16<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi32_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtepi32_storeu_epi8<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi64_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtepi64_storeu_epi16<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi64_storeu_epi32)
#[inline(always)]
pub fn _mm512_mask_cvtepi64_storeu_epi32<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi64_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtepi64_storeu_epi8<T: Is64BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi32_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtsepi32_storeu_epi16<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi32_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtsepi32_storeu_epi8<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi64_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtsepi64_storeu_epi16<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi64_storeu_epi32)
#[inline(always)]
pub fn _mm512_mask_cvtsepi64_storeu_epi32<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi64_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtsepi64_storeu_epi8<T: Is64BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi32_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtusepi32_storeu_epi16<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi32_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtusepi32_storeu_epi8<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi64_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_cvtusepi64_storeu_epi16<T: Is128BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 32-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi64_storeu_epi32)
#[inline(always)]
pub fn _mm512_mask_cvtusepi64_storeu_epi32<T: Is256BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi64_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtusepi64_storeu_epi8<T: Is64BitsUnaligned>(_token: Avx512fToken, base_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi32)
#[inline(always)]
pub fn _mm512_mask_storeu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: &mut T, k: __mmask16, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_epi32::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 512-bits (composed of 16 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi32)
#[inline(always)]
pub fn _mm512_storeu_epi32<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: &mut T, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_storeu_epi32::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi64)
#[inline(always)]
pub fn _mm512_mask_storeu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: &mut T, k: __mmask8, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_epi64::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 512-bits (composed of 8 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi64)
#[inline(always)]
pub fn _mm512_storeu_epi64<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: &mut T, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_storeu_epi64::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_pd)
#[inline(always)]
pub fn _mm512_mask_storeu_pd(_token: Avx512fToken, mem_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_pd(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Stores 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_pd)
#[inline(always)]
pub fn _mm512_storeu_pd(_token: Avx512fToken, mem_addr: &mut [f64; 8], a: __m512d) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &mut [f64; 8], a: __m512d) {
        safe_unaligned_simd::x86_64::_mm512_storeu_pd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_ps)
#[inline(always)]
pub fn _mm512_mask_storeu_ps(_token: Avx512fToken, mem_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_ps(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Stores 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_ps)
#[inline(always)]
pub fn _mm512_storeu_ps(_token: Avx512fToken, mem_addr: &mut [f32; 16], a: __m512) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner(mem_addr: &mut [f32; 16], a: __m512) {
        safe_unaligned_simd::x86_64::_mm512_storeu_ps(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store 512-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_si512)
#[inline(always)]
pub fn _mm512_storeu_si512<T: Is512BitsUnaligned>(_token: Avx512fToken, mem_addr: &mut T, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_storeu_si512::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

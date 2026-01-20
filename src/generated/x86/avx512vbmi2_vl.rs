//! Token-gated wrappers for `#[target_feature(enable = "avx512vbmi2,avx512vl")]` functions.
//!
//! This module contains 12 functions that are safe to call when you have a [`Avx512Vbmi2VlToken`].
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
    Is16BitsUnaligned, Is16CellUnaligned, Is32BitsUnaligned, Is32CellUnaligned, Is64BitsUnaligned,
    Is64CellUnaligned, Is128BitsUnaligned, Is128CellUnaligned, Is256BitsUnaligned,
    Is256CellUnaligned, Is512BitsUnaligned,
};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    Is16BitsUnaligned, Is16CellUnaligned, Is32BitsUnaligned, Is32CellUnaligned, Is64BitsUnaligned,
    Is64CellUnaligned, Is128BitsUnaligned, Is128CellUnaligned, Is256BitsUnaligned,
    Is256CellUnaligned, Is512BitsUnaligned,
};

use crate::tokens::x86::Avx512Vbmi2VlToken;

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi16)
#[inline(always)]
pub fn _mm_mask_expandloadu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi16)
#[inline(always)]
pub fn _mm_maskz_expandloadu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi16)
#[inline(always)]
pub fn _mm256_mask_expandloadu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    src: __m256i,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask16, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi16)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi8)
#[inline(always)]
pub fn _mm_mask_expandloadu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    src: __m128i,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask16, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi8)
#[inline(always)]
pub fn _mm_maskz_expandloadu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi8)
#[inline(always)]
pub fn _mm256_mask_expandloadu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    src: __m256i,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask32, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi8)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi16)
#[inline(always)]
pub fn _mm_mask_compressstoreu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi16)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi8)
#[inline(always)]
pub fn _mm_mask_compressstoreu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    base_addr: &mut T,
    k: __mmask16,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi8)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512Vbmi2VlToken,
    base_addr: &mut T,
    k: __mmask32,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask32, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

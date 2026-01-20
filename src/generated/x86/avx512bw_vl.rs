//! Token-gated wrappers for `#[target_feature(enable = "avx512bw,avx512vl")]` functions.
//!
//! This module contains 26 functions that are safe to call when you have a [`Avx512bwVlToken`].
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

use crate::tokens::x86::Avx512bwVlToken;

/// Load 128-bits (composed of 8 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi16)
#[inline(always)]
pub fn _mm_loadu_epi16<T: Is128BitsUnaligned>(_token: Avx512bwVlToken, mem_addr: &T) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_epi16::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi16)
#[inline(always)]
pub fn _mm_mask_loadu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi16)
#[inline(always)]
pub fn _mm_maskz_loadu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 256-bits (composed of 16 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi16)
#[inline(always)]
pub fn _mm256_loadu_epi16<T: Is256BitsUnaligned>(_token: Avx512bwVlToken, mem_addr: &T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu_epi16::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi16)
#[inline(always)]
pub fn _mm256_mask_loadu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    src: __m256i,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask16, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi16)
#[inline(always)]
pub fn _mm256_maskz_loadu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 128-bits (composed of 16 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi8)
#[inline(always)]
pub fn _mm_loadu_epi8<T: Is128BitsUnaligned>(_token: Avx512bwVlToken, mem_addr: &T) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_epi8::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi8)
#[inline(always)]
pub fn _mm_mask_loadu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    src: __m128i,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask16, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi8)
#[inline(always)]
pub fn _mm_maskz_loadu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 256-bits (composed of 32 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi8)
#[inline(always)]
pub fn _mm256_loadu_epi8<T: Is256BitsUnaligned>(_token: Avx512bwVlToken, mem_addr: &T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu_epi8::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi8)
#[inline(always)]
pub fn _mm256_mask_loadu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    src: __m256i,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask32, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi8)
#[inline(always)]
pub fn _mm256_maskz_loadu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi16_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtepi16_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi16_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtepi16_storeu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi16_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtsepi16_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi16_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtsepi16_storeu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi16_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtusepi16_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi16_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtusepi16_storeu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask16, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_storeu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_epi16::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 128-bits (composed of 8 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi16)
#[inline(always)]
pub fn _mm_storeu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_epi16::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_storeu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_epi16::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 256-bits (composed of 16 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi16)
#[inline(always)]
pub fn _mm256_storeu_epi16<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu_epi16::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_storeu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    k: __mmask16,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_epi8::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 128-bits (composed of 16 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi8)
#[inline(always)]
pub fn _mm_storeu_epi8<T: Is128BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_epi8::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_storeu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    k: __mmask32,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_epi8::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 256-bits (composed of 32 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi8)
#[inline(always)]
pub fn _mm256_storeu_epi8<T: Is256BitsUnaligned>(
    _token: Avx512bwVlToken,
    mem_addr: &mut T,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu_epi8::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

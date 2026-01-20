//! Token-gated wrappers for `#[target_feature(enable = "avx512f,avx512vl")]` functions.
//!
//! This module contains 86 functions that are safe to call when you have a [`Avx512fVlToken`].
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

use crate::tokens::x86::Avx512fVlToken;

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi32)
#[inline(always)]
pub fn _mm_mask_expandloadu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi32)
#[inline(always)]
pub fn _mm_maskz_expandloadu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi32)
#[inline(always)]
pub fn _mm256_mask_expandloadu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi32)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi64)
#[inline(always)]
pub fn _mm_mask_expandloadu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi64)
#[inline(always)]
pub fn _mm_maskz_expandloadu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi64)
#[inline(always)]
pub fn _mm256_mask_expandloadu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi64)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_pd)
#[inline(always)]
pub fn _mm_mask_expandloadu_pd(
    _token: Avx512fVlToken,
    src: __m128d,
    k: __mmask8,
    mem_addr: &[f64; 2],
) -> __m128d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_pd)
#[inline(always)]
pub fn _mm_maskz_expandloadu_pd(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &[f64; 2],
) -> __m128d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_pd)
#[inline(always)]
pub fn _mm256_mask_expandloadu_pd(
    _token: Avx512fVlToken,
    src: __m256d,
    k: __mmask8,
    mem_addr: &[f64; 4],
) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_pd)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_pd(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &[f64; 4],
) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_ps)
#[inline(always)]
pub fn _mm_mask_expandloadu_ps(
    _token: Avx512fVlToken,
    src: __m128,
    k: __mmask8,
    mem_addr: &[f32; 4],
) -> __m128 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_mask_expandloadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_ps)
#[inline(always)]
pub fn _mm_maskz_expandloadu_ps(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &[f32; 4],
) -> __m128 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_maskz_expandloadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_ps)
#[inline(always)]
pub fn _mm256_mask_expandloadu_ps(
    _token: Avx512fVlToken,
    src: __m256,
    k: __mmask8,
    mem_addr: &[f32; 8],
) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_mask_expandloadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_ps)
#[inline(always)]
pub fn _mm256_maskz_expandloadu_ps(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &[f32; 8],
) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_maskz_expandloadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 128-bits (composed of 4 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi32)
#[inline(always)]
pub fn _mm_loadu_epi32<T: Is128BitsUnaligned>(_token: Avx512fVlToken, mem_addr: &T) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_epi32::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi32)
#[inline(always)]
pub fn _mm_mask_loadu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi32)
#[inline(always)]
pub fn _mm_maskz_loadu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 256-bits (composed of 8 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi32)
#[inline(always)]
pub fn _mm256_loadu_epi32<T: Is256BitsUnaligned>(_token: Avx512fVlToken, mem_addr: &T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu_epi32::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi32)
#[inline(always)]
pub fn _mm256_mask_loadu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_epi32::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi32)
#[inline(always)]
pub fn _mm256_maskz_loadu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_epi32::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 128-bits (composed of 2 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi64)
#[inline(always)]
pub fn _mm_loadu_epi64<T: Is128BitsUnaligned>(_token: Avx512fVlToken, mem_addr: &T) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_epi64::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi64)
#[inline(always)]
pub fn _mm_mask_loadu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(src: __m128i, k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi64)
#[inline(always)]
pub fn _mm_maskz_loadu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 256-bits (composed of 4 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi64)
#[inline(always)]
pub fn _mm256_loadu_epi64<T: Is256BitsUnaligned>(_token: Avx512fVlToken, mem_addr: &T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu_epi64::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi64)
#[inline(always)]
pub fn _mm256_mask_loadu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(src: __m256i, k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_epi64::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi64)
#[inline(always)]
pub fn _mm256_maskz_loadu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_epi64::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_pd)
#[inline(always)]
pub fn _mm_mask_loadu_pd(
    _token: Avx512fVlToken,
    src: __m128d,
    k: __mmask8,
    mem_addr: &[f64; 2],
) -> __m128d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_pd)
#[inline(always)]
pub fn _mm_maskz_loadu_pd(_token: Avx512fVlToken, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_pd)
#[inline(always)]
pub fn _mm256_mask_loadu_pd(
    _token: Avx512fVlToken,
    src: __m256d,
    k: __mmask8,
    mem_addr: &[f64; 4],
) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_pd(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_pd)
#[inline(always)]
pub fn _mm256_maskz_loadu_pd(_token: Avx512fVlToken, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_pd(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_ps)
#[inline(always)]
pub fn _mm_mask_loadu_ps(
    _token: Avx512fVlToken,
    src: __m128,
    k: __mmask8,
    mem_addr: &[f32; 4],
) -> __m128 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_mask_loadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_ps)
#[inline(always)]
pub fn _mm_maskz_loadu_ps(_token: Avx512fVlToken, k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_maskz_loadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_ps)
#[inline(always)]
pub fn _mm256_mask_loadu_ps(
    _token: Avx512fVlToken,
    src: __m256,
    k: __mmask8,
    mem_addr: &[f32; 8],
) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_mask_loadu_ps(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_ps)
#[inline(always)]
pub fn _mm256_maskz_loadu_ps(_token: Avx512fVlToken, k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_maskz_loadu_ps(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi32)
#[inline(always)]
pub fn _mm_mask_compressstoreu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi32)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi64)
#[inline(always)]
pub fn _mm_mask_compressstoreu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_epi64::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi64)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_epi64::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_pd)
#[inline(always)]
pub fn _mm_mask_compressstoreu_pd(
    _token: Avx512fVlToken,
    base_addr: &mut [f64; 2],
    k: __mmask8,
    a: __m128d,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(base_addr: &mut [f64; 2], k: __mmask8, a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_pd(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_pd)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_pd(
    _token: Avx512fVlToken,
    base_addr: &mut [f64; 4],
    k: __mmask8,
    a: __m256d,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(base_addr: &mut [f64; 4], k: __mmask8, a: __m256d) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_pd(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_ps)
#[inline(always)]
pub fn _mm_mask_compressstoreu_ps(
    _token: Avx512fVlToken,
    base_addr: &mut [f32; 4],
    k: __mmask8,
    a: __m128,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(base_addr: &mut [f32; 4], k: __mmask8, a: __m128) {
        safe_unaligned_simd::x86_64::_mm_mask_compressstoreu_ps(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_ps)
#[inline(always)]
pub fn _mm256_mask_compressstoreu_ps(
    _token: Avx512fVlToken,
    base_addr: &mut [f32; 8],
    k: __mmask8,
    a: __m256,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(base_addr: &mut [f32; 8], k: __mmask8, a: __m256) {
        safe_unaligned_simd::x86_64::_mm256_mask_compressstoreu_ps(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi32_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtepi32_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi32_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtepi32_storeu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi32_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtepi32_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi32_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtepi32_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi64_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtepi64_storeu_epi16<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi64_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtepi64_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi64_storeu_epi32)
#[inline(always)]
pub fn _mm_mask_cvtepi64_storeu_epi32<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi64_storeu_epi32)
#[inline(always)]
pub fn _mm256_mask_cvtepi64_storeu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi64_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtepi64_storeu_epi8<T: Is16BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is16BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi64_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtepi64_storeu_epi8<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi32_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtsepi32_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi32_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtsepi32_storeu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi32_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtsepi32_storeu_epi8<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi32_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtsepi32_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi64_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtsepi64_storeu_epi16<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi64_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtsepi64_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi64_storeu_epi32)
#[inline(always)]
pub fn _mm_mask_cvtsepi64_storeu_epi32<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi64_storeu_epi32)
#[inline(always)]
pub fn _mm256_mask_cvtsepi64_storeu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi64_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtsepi64_storeu_epi8<T: Is16BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is16BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtsepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi64_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtsepi64_storeu_epi8<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtsepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi32_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtusepi32_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi32_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtusepi32_storeu_epi16<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi32_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi32_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtusepi32_storeu_epi8<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 32-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi32_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtusepi32_storeu_epi8<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi32_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi64_storeu_epi16)
#[inline(always)]
pub fn _mm_mask_cvtusepi64_storeu_epi16<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 16-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi64_storeu_epi16)
#[inline(always)]
pub fn _mm256_mask_cvtusepi64_storeu_epi16<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi64_storeu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 32-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi64_storeu_epi32)
#[inline(always)]
pub fn _mm_mask_cvtusepi64_storeu_epi32<T: Is64BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is64BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 32-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi64_storeu_epi32)
#[inline(always)]
pub fn _mm256_mask_cvtusepi64_storeu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi64_storeu_epi32::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi64_storeu_epi8)
#[inline(always)]
pub fn _mm_mask_cvtusepi64_storeu_epi8<T: Is16BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is16BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_cvtusepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 64-bit integers in a to packed 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi64_storeu_epi8)
#[inline(always)]
pub fn _mm256_mask_cvtusepi64_storeu_epi8<T: Is32BitsUnaligned>(
    _token: Avx512fVlToken,
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is32BitsUnaligned>(base_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_cvtusepi64_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi32)
#[inline(always)]
pub fn _mm_mask_storeu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_epi32::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 128-bits (composed of 4 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi32)
#[inline(always)]
pub fn _mm_storeu_epi32<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_epi32::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi32)
#[inline(always)]
pub fn _mm256_mask_storeu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_epi32::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 256-bits (composed of 8 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi32)
#[inline(always)]
pub fn _mm256_storeu_epi32<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu_epi32::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi64)
#[inline(always)]
pub fn _mm_mask_storeu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_epi64::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 128-bits (composed of 2 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi64)
#[inline(always)]
pub fn _mm_storeu_epi64<T: Is128BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    a: __m128i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_epi64::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi64)
#[inline(always)]
pub fn _mm256_mask_storeu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_epi64::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 256-bits (composed of 4 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi64)
#[inline(always)]
pub fn _mm256_storeu_epi64<T: Is256BitsUnaligned>(
    _token: Avx512fVlToken,
    mem_addr: &mut T,
    a: __m256i,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu_epi64::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_pd)
#[inline(always)]
pub fn _mm_mask_storeu_pd(
    _token: Avx512fVlToken,
    mem_addr: &mut [f64; 2],
    k: __mmask8,
    a: __m128d,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(mem_addr: &mut [f64; 2], k: __mmask8, a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_pd(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_pd)
#[inline(always)]
pub fn _mm256_mask_storeu_pd(
    _token: Avx512fVlToken,
    mem_addr: &mut [f64; 4],
    k: __mmask8,
    a: __m256d,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(mem_addr: &mut [f64; 4], k: __mmask8, a: __m256d) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_pd(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_ps)
#[inline(always)]
pub fn _mm_mask_storeu_ps(_token: Avx512fVlToken, mem_addr: &mut [f32; 4], k: __mmask8, a: __m128) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(mem_addr: &mut [f32; 4], k: __mmask8, a: __m128) {
        safe_unaligned_simd::x86_64::_mm_mask_storeu_ps(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_ps)
#[inline(always)]
pub fn _mm256_mask_storeu_ps(
    _token: Avx512fVlToken,
    mem_addr: &mut [f32; 8],
    k: __mmask8,
    a: __m256,
) {
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn inner(mem_addr: &mut [f32; 8], k: __mmask8, a: __m256) {
        safe_unaligned_simd::x86_64::_mm256_mask_storeu_ps(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

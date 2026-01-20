//! Token-gated wrappers for `#[target_feature(enable = "avx512bw")]` functions.
//!
//! This module contains 13 functions that are safe to call when you have a [`Avx512bwToken`].
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

use crate::tokens::x86::Avx512bwToken;

/// Load 512-bits (composed of 32 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi16)
#[inline(always)]
pub fn _mm512_loadu_epi16<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_loadu_epi16::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi16)
#[inline(always)]
pub fn _mm512_mask_loadu_epi16<T: Is512BitsUnaligned>(_token: Avx512bwToken, src: __m512i, k: __mmask32, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask32, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi16)
#[inline(always)]
pub fn _mm512_maskz_loadu_epi16<T: Is512BitsUnaligned>(_token: Avx512bwToken, k: __mmask32, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask32, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load 512-bits (composed of 64 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi8)
#[inline(always)]
pub fn _mm512_loadu_epi8<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_loadu_epi8::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi8)
#[inline(always)]
pub fn _mm512_mask_loadu_epi8<T: Is512BitsUnaligned>(_token: Avx512bwToken, src: __m512i, k: __mmask64, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask64, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_loadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi8)
#[inline(always)]
pub fn _mm512_maskz_loadu_epi8<T: Is512BitsUnaligned>(_token: Avx512bwToken, k: __mmask64, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_loadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi16_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtepi16_storeu_epi8<T: Is256BitsUnaligned>(_token: Avx512bwToken, base_addr: &mut T, k: __mmask32, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask32, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi16_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtsepi16_storeu_epi8<T: Is256BitsUnaligned>(_token: Avx512bwToken, base_addr: &mut T, k: __mmask32, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask32, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtsepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi16_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_cvtusepi16_storeu_epi8<T: Is256BitsUnaligned>(_token: Avx512bwToken, base_addr: &mut T, k: __mmask32, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is256BitsUnaligned>(base_addr: &mut T, k: __mmask32, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_cvtusepi16_storeu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi16)
#[inline(always)]
pub fn _mm512_mask_storeu_epi16<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: &mut T, k: __mmask32, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_epi16::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 512-bits (composed of 32 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi16)
#[inline(always)]
pub fn _mm512_storeu_epi16<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: &mut T, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_storeu_epi16::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi8)
#[inline(always)]
pub fn _mm512_mask_storeu_epi8<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: &mut T, k: __mmask64, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask64, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_storeu_epi8::<T>(mem_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, k, a) }
}

/// Store 512-bits (composed of 64 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi8)
#[inline(always)]
pub fn _mm512_storeu_epi8<T: Is512BitsUnaligned>(_token: Avx512bwToken, mem_addr: &mut T, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn inner<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_storeu_epi8::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

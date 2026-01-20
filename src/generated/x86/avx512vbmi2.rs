//! Token-gated wrappers for `#[target_feature(enable = "avx512vbmi2")]` functions.
//!
//! This module contains 6 functions that are safe to call when you have a [`HasAvx512vbmi2`].
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

use crate::tokens::{HasAvx512vbmi2};

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi16)
#[inline(always)]
pub fn _mm512_mask_expandloadu_epi16<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, src: __m512i, k: __mmask32, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask32, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_epi16::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi16)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_epi16<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, k: __mmask32, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask32, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_epi16::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi8)
#[inline(always)]
pub fn _mm512_mask_expandloadu_epi8<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, src: __m512i, k: __mmask64, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(src: __m512i, k: __mmask64, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_mask_expandloadu_epi8::<T>(src, k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(src, k, mem_addr) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi8)
#[inline(always)]
pub fn _mm512_maskz_expandloadu_epi8<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, k: __mmask64, mem_addr: & T) -> __m512i {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: & T) -> __m512i {
        safe_unaligned_simd::x86_64::_mm512_maskz_expandloadu_epi8::<T>(k, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(k, mem_addr) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi16)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_epi16<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, base_addr: &mut T, k: __mmask32, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(base_addr: &mut T, k: __mmask32, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_epi16::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi8)
#[inline(always)]
pub fn _mm512_mask_compressstoreu_epi8<T: Is512BitsUnaligned>(_token: impl HasAvx512vbmi2, base_addr: &mut T, k: __mmask64, a: __m512i) {
    #[inline]
    #[target_feature(enable = "avx512vbmi2")]
    unsafe fn inner<T: Is512BitsUnaligned>(base_addr: &mut T, k: __mmask64, a: __m512i) {
        safe_unaligned_simd::x86_64::_mm512_mask_compressstoreu_epi8::<T>(base_addr, k, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(base_addr, k, a) }
}

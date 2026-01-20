//! Token-gated wrappers for `#[target_feature(enable = "avx")]` functions.
//!
//! This module contains 17 functions that are safe to call when you have a [`HasAvx`].
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

use crate::tokens::{HasAvx};

/// Broadcasts 128 bits from memory (composed of 2 packed double-precision
/// (64-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_pd)
#[inline(always)]
pub fn _mm256_broadcast_pd(_token: impl HasAvx, mem_addr: & __m128d) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: & __m128d) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_broadcast_pd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Broadcasts 128 bits from memory (composed of 4 packed single-precision
/// (32-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_ps)
#[inline(always)]
pub fn _mm256_broadcast_ps(_token: impl HasAvx, mem_addr: & __m128) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: & __m128) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_broadcast_ps(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Broadcasts a double-precision (64-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_sd)
#[inline(always)]
pub fn _mm256_broadcast_sd(_token: impl HasAvx, mem_addr: & f64) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: & f64) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_broadcast_sd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcast_ss)
#[inline(always)]
pub fn _mm_broadcast_ss(_token: impl HasAvx, mem_addr: & f32) -> __m128 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: & f32) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_broadcast_ss(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_ss)
#[inline(always)]
pub fn _mm256_broadcast_ss(_token: impl HasAvx, mem_addr: & f32) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: & f32) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_broadcast_ss(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_pd)
#[inline(always)]
pub fn _mm256_loadu_pd(_token: impl HasAvx, mem_addr: &[f64; 4]) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: &[f64; 4]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_loadu_pd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_ps)
#[inline(always)]
pub fn _mm256_loadu_ps(_token: impl HasAvx, mem_addr: &[f32; 8]) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: &[f32; 8]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_loadu_ps(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads 256-bits of integer data from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_si256)
#[inline(always)]
pub fn _mm256_loadu_si256<T: Is256BitsUnaligned>(_token: impl HasAvx, mem_addr: & T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: & T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu_si256::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads two 128-bit values (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128)
#[inline(always)]
pub fn _mm256_loadu2_m128(_token: impl HasAvx, hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256 {
        safe_unaligned_simd::x86_64::_mm256_loadu2_m128(hiaddr, loaddr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr) }
}

/// Loads two 128-bit values (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128d)
#[inline(always)]
pub fn _mm256_loadu2_m128d(_token: impl HasAvx, hiaddr: &[f64; 2], loaddr: &[f64; 2]) -> __m256d {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(hiaddr: &[f64; 2], loaddr: &[f64; 2]) -> __m256d {
        safe_unaligned_simd::x86_64::_mm256_loadu2_m128d(hiaddr, loaddr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr) }
}

/// Loads two 128-bit values (composed of integer data) from memory, and combine
/// them into a 256-bit value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128i)
#[inline(always)]
pub fn _mm256_loadu2_m128i<T: Is128BitsUnaligned>(_token: impl HasAvx, hiaddr: & T, loaddr: & T) -> __m256i {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner<T: Is128BitsUnaligned>(hiaddr: & T, loaddr: & T) -> __m256i {
        safe_unaligned_simd::x86_64::_mm256_loadu2_m128i::<T>(hiaddr, loaddr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr) }
}

/// Stores 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_pd)
#[inline(always)]
pub fn _mm256_storeu_pd(_token: impl HasAvx, mem_addr: &mut [f64; 4], a: __m256d) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: &mut [f64; 4], a: __m256d) {
        safe_unaligned_simd::x86_64::_mm256_storeu_pd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps)
#[inline(always)]
pub fn _mm256_storeu_ps(_token: impl HasAvx, mem_addr: &mut [f32; 8], a: __m256) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(mem_addr: &mut [f32; 8], a: __m256) {
        safe_unaligned_simd::x86_64::_mm256_storeu_ps(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores 256-bits of integer data from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_si256)
#[inline(always)]
pub fn _mm256_storeu_si256<T: Is256BitsUnaligned>(_token: impl HasAvx, mem_addr: &mut T, a: __m256i) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu_si256::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores the high and low 128-bit halves (each composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128)
#[inline(always)]
pub fn _mm256_storeu2_m128(_token: impl HasAvx, hiaddr: &mut [f32; 4], loaddr: &mut [f32; 4], a: __m256) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(hiaddr: &mut [f32; 4], loaddr: &mut [f32; 4], a: __m256) {
        safe_unaligned_simd::x86_64::_mm256_storeu2_m128(hiaddr, loaddr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr, a) }
}

/// Stores the high and low 128-bit halves (each composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128d)
#[inline(always)]
pub fn _mm256_storeu2_m128d(_token: impl HasAvx, hiaddr: &mut [f64; 2], loaddr: &mut [f64; 2], a: __m256d) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner(hiaddr: &mut [f64; 2], loaddr: &mut [f64; 2], a: __m256d) {
        safe_unaligned_simd::x86_64::_mm256_storeu2_m128d(hiaddr, loaddr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr, a) }
}

/// Stores the high and low 128-bit halves (each composed of integer data) from
/// `a` into memory two different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128i)
#[inline(always)]
pub fn _mm256_storeu2_m128i<T: Is128BitsUnaligned>(_token: impl HasAvx, hiaddr: &mut T, loaddr: &mut T, a: __m256i) {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn inner<T: Is128BitsUnaligned>(hiaddr: &mut T, loaddr: &mut T, a: __m256i) {
        safe_unaligned_simd::x86_64::_mm256_storeu2_m128i::<T>(hiaddr, loaddr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(hiaddr, loaddr, a) }
}

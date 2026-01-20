//! Token-gated wrappers for `#[target_feature(enable = "sse2")]` functions.
//!
//! This module contains 20 functions that are safe to call when you have a [`HasSse2`].
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

use crate::tokens::{HasSse2};

/// Loads a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_pd1)
#[inline(always)]
pub fn _mm_load_pd1(_token: impl HasSse2, mem_addr: & f64) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: & f64) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_load_pd1(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads a 64-bit double-precision value to the low element of a
/// 128-bit integer vector and clears the upper element.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_sd)
#[inline(always)]
pub fn _mm_load_sd(_token: impl HasSse2, mem_addr: & f64) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: & f64) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_load_sd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_pd)
#[inline(always)]
pub fn _mm_load1_pd(_token: impl HasSse2, mem_addr: & f64) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: & f64) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_load1_pd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads a double-precision value into the high-order bits of a 128-bit
/// vector of `[2 x double]`. The low-order bits are copied from the low-order
/// bits of the first operand.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadh_pd)
#[inline(always)]
pub fn _mm_loadh_pd(_token: impl HasSse2, a: __m128d, mem_addr: & f64) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(a: __m128d, mem_addr: & f64) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_loadh_pd(a, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(a, mem_addr) }
}

/// Loads a 64-bit integer from memory into first element of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64)
#[inline(always)]
pub fn _mm_loadl_epi64<T: Is128BitsUnaligned>(_token: impl HasSse2, mem_addr: & T) -> __m128i {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: & T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadl_epi64::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads a double-precision value into the low-order bits of a 128-bit
/// vector of `[2 x double]`. The high-order bits are copied from the
/// high-order bits of the first operand.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_pd)
#[inline(always)]
pub fn _mm_loadl_pd(_token: impl HasSse2, a: __m128d, mem_addr: & f64) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(a: __m128d, mem_addr: & f64) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_loadl_pd(a, mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(a, mem_addr) }
}

/// Loads 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_pd)
#[inline(always)]
pub fn _mm_loadu_pd(_token: impl HasSse2, mem_addr: &[f64; 2]) -> __m128d {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: &[f64; 2]) -> __m128d {
        safe_unaligned_simd::x86_64::_mm_loadu_pd(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads 128-bits of integer data from memory into a new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si128)
#[inline(always)]
pub fn _mm_loadu_si128<T: Is128BitsUnaligned>(_token: impl HasSse2, mem_addr: & T) -> __m128i {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: & T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_si128::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads unaligned 16-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si16)
#[inline(always)]
pub fn _mm_loadu_si16<T: Is16BitsUnaligned>(_token: impl HasSse2, mem_addr: & T) -> __m128i {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is16BitsUnaligned>(mem_addr: & T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_si16::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads unaligned 32-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si32)
#[inline(always)]
pub fn _mm_loadu_si32<T: Is32BitsUnaligned>(_token: impl HasSse2, mem_addr: & T) -> __m128i {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is32BitsUnaligned>(mem_addr: & T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_si32::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads unaligned 64-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si64)
#[inline(always)]
pub fn _mm_loadu_si64<T: Is64BitsUnaligned>(_token: impl HasSse2, mem_addr: & T) -> __m128i {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is64BitsUnaligned>(mem_addr: & T) -> __m128i {
        safe_unaligned_simd::x86_64::_mm_loadu_si64::<T>(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Stores the lower 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_sd)
#[inline(always)]
pub fn _mm_store_sd(_token: impl HasSse2, mem_addr: &mut f64, a: __m128d) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: &mut f64, a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_store_sd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores the upper 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeh_pd)
#[inline(always)]
pub fn _mm_storeh_pd(_token: impl HasSse2, mem_addr: &mut f64, a: __m128d) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: &mut f64, a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_storeh_pd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores the lower 64-bit integer `a` to a memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_epi64)
#[inline(always)]
pub fn _mm_storel_epi64<T: Is128BitsUnaligned>(_token: impl HasSse2, mem_addr: &mut T, a: __m128i) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storel_epi64::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores the lower 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_pd)
#[inline(always)]
pub fn _mm_storel_pd(_token: impl HasSse2, mem_addr: &mut f64, a: __m128d) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: &mut f64, a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_storel_pd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_pd)
#[inline(always)]
pub fn _mm_storeu_pd(_token: impl HasSse2, mem_addr: &mut [f64; 2], a: __m128d) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner(mem_addr: &mut [f64; 2], a: __m128d) {
        safe_unaligned_simd::x86_64::_mm_storeu_pd(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores 128-bits of integer data from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si128)
#[inline(always)]
pub fn _mm_storeu_si128<T: Is128BitsUnaligned>(_token: impl HasSse2, mem_addr: &mut T, a: __m128i) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_si128::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store 16-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si16)
#[inline(always)]
pub fn _mm_storeu_si16<T: Is16BitsUnaligned>(_token: impl HasSse2, mem_addr: &mut T, a: __m128i) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is16BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_si16::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store 32-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si32)
#[inline(always)]
pub fn _mm_storeu_si32<T: Is32BitsUnaligned>(_token: impl HasSse2, mem_addr: &mut T, a: __m128i) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is32BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_si32::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Store 64-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si64)
#[inline(always)]
pub fn _mm_storeu_si64<T: Is64BitsUnaligned>(_token: impl HasSse2, mem_addr: &mut T, a: __m128i) {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn inner<T: Is64BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
        safe_unaligned_simd::x86_64::_mm_storeu_si64::<T>(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

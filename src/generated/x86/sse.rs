//! Token-gated wrappers for `#[target_feature(enable = "sse")]` functions.
//!
//! This module contains 6 functions that are safe to call when you have a [`HasSse42`].
//! Note: SSE4.2 is the baseline for archmage, so HasSse42 is used instead of HasSse.
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

use crate::tokens::HasSse42;

/// Construct a [`__m128`] by duplicating the value read from `mem_addr` into
/// all elements.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS` followed by some
/// shuffling.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_ps)
#[inline(always)]
pub fn _mm_load1_ps(_token: impl HasSse42, mem_addr: &f32) -> __m128 {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &f32) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_load1_ps(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Alias for [`_mm_load1_ps`].
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ps1)
#[inline(always)]
pub fn _mm_load_ps1(_token: impl HasSse42, mem_addr: &f32) -> __m128 {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &f32) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_load_ps1(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Construct a [`__m128`] with the lowest element read from `mem_addr` and the
/// other elements set to zero.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ss)
#[inline(always)]
pub fn _mm_load_ss(_token: impl HasSse42, mem_addr: &f32) -> __m128 {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &f32) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_load_ss(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Loads four `f32` values from memory into a [`__m128`]. There are no
/// restrictions on memory alignment.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_ps)
#[inline(always)]
pub fn _mm_loadu_ps(_token: impl HasSse42, mem_addr: &[f32; 4]) -> __m128 {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &[f32; 4]) -> __m128 {
        safe_unaligned_simd::x86_64::_mm_loadu_ps(mem_addr)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr) }
}

/// Stores the lowest 32-bit float of `a` into memory.
///
/// This intrinsic corresponds to the `MOVSS` instruction.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_ss)
#[inline(always)]
pub fn _mm_store_ss(_token: impl HasSse42, mem_addr: &mut f32, a: __m128) {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &mut f32, a: __m128) {
        safe_unaligned_simd::x86_64::_mm_store_ss(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

/// Stores four 32-bit floats into memory. There are no restrictions on memory
/// alignment.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_ps)
#[inline(always)]
pub fn _mm_storeu_ps(_token: impl HasSse42, mem_addr: &mut [f32; 4], a: __m128) {
    #[inline]
    #[target_feature(enable = "sse")]
    unsafe fn inner(mem_addr: &mut [f32; 4], a: __m128) {
        safe_unaligned_simd::x86_64::_mm_storeu_ps(mem_addr, a)
    }
    // SAFETY: Token proves the target features are available
    unsafe { inner(mem_addr, a) }
}

//! Bounds-safe AVX-512 gather and scatter for the native 16-lane 32-bit
//! types: `u32x16`, `i32x16` and `f32x16` on `X64V4Token`. An `X64V4xToken`
//! converts with `.v4()`.
//!
//! | Method | Lane `i` | Index out of range |
//! |---|---|---|
//! | `T::gather_wrapping(&[E; N], idx)` | `table[idx[i] & (N - 1)]` | wraps; `N` is a power of two |
//! | `T::gather_or(&[E], idx, or)` | `table[idx[i]]` | lane keeps `or[i]` |
//! | `v.scatter_select(&mut [E], enable, idx)` | `dst[idx[i]] = v[i]` if bit `i` of `enable` is set | write skipped |
//!
//! Indices are `u32x16` lanes read as unsigned. Indices at or above 2^31 are
//! out of range even when the slice is longer, because the instructions use
//! signed 32-bit offsets. Scatter writes lanes in order, so when lanes share
//! an index the highest lane's value is the one left in memory.
//!
//! Nothing here exists for other widths or backends. A hardware gather is not
//! reliably faster than per-lane scalar loads, and the balance changes from
//! one CPU generation to the next. Elsewhere, build the vector from scalar
//! lookups with `from_array_t`, and measure before choosing either form.
//!
//! # Safety argument
//!
//! The gather and scatter intrinsics take a base pointer and per-lane offsets,
//! so they stay `unsafe` inside an AVX-512 region. Each call below meets one
//! condition for every lane the instruction accesses: its offset `o`
//! satisfies `0 <= o < len`, so the 4-byte access at `base + 4 * o` lies
//! inside the borrowed slice.
//!
//! - Wrapping gathers mask the offsets to `idx & (N - 1)`, where `N` is a power
//!   of two no larger than 2^31 (a compile-time assert). Every offset is in
//!   `0..N`.
//! - Slice gathers and scatters enable a lane only if its index is below
//!   `min(len, 2^31)` as an unsigned value. Masked-off lanes access no memory,
//!   so an empty slice is fine. Enabled offsets are below 2^31, so sign
//!   extension cannot make them negative.
//! - Scatters write through `&mut`, so nothing else sees the memory during the
//!   call. `u32`, `i32` and `f32` accept every bit pattern.
//! - The `X64V4Token` stored in the index (or value) vector proves AVX-512F,
//!   and every helper is an `#[arcane]` region for that token.
//!
//! `cargo xtask soundness` rejects gather and scatter intrinsics anywhere else
//! in magetypes, so no call can skip these checks.

use super::{f32x16, i32x16, u32x16};
use archmage::X64V4Token;
use core::arch::x86_64::{__m512, __m512i};

/// Table element types read and written through the integer instructions.
trait Int32: Copy {}
impl Int32 for u32 {}
impl Int32 for i32 {}

const fn assert_wrapping_table<const N: usize>() {
    assert!(
        N.is_power_of_two() && N <= 1 << 31,
        "gather_wrapping: the table length must be a power of two no larger than 2^31"
    );
}

/// Exclusive bound on enabled indices for a slice of `len` elements:
/// `min(len, 2^31)`, as the bit pattern of the unsigned compare operand.
#[inline(always)]
fn lane_bound(len: usize) -> i32 {
    len.min(1 << 31) as u32 as i32
}

#[archmage::arcane(import_intrinsics)]
fn gather_wrapping_epi32<E: Int32, const N: usize>(
    _token: X64V4Token,
    table: &[E; N],
    idx: __m512i,
) -> __m512i {
    const { assert_wrapping_table::<N>() };
    let off = _mm512_and_si512(idx, _mm512_set1_epi32((N - 1) as i32));
    // SAFETY: every offset is `idx & (N - 1)`, in `0..N` because `N` is a
    // power of two no larger than 2^31, so each lane reads 4 bytes of `table`.
    unsafe { _mm512_i32gather_epi32::<4>(off, table.as_ptr().cast()) }
}

#[archmage::arcane(import_intrinsics)]
fn gather_wrapping_ps<const N: usize>(
    _token: X64V4Token,
    table: &[f32; N],
    idx: __m512i,
) -> __m512 {
    const { assert_wrapping_table::<N>() };
    let off = _mm512_and_si512(idx, _mm512_set1_epi32((N - 1) as i32));
    // SAFETY: every offset is `idx & (N - 1)`, in `0..N` because `N` is a
    // power of two no larger than 2^31, so each lane reads 4 bytes of `table`.
    unsafe { _mm512_i32gather_ps::<4>(off, table.as_ptr()) }
}

#[archmage::arcane(import_intrinsics)]
fn gather_or_epi32<E: Int32>(
    _token: X64V4Token,
    table: &[E],
    idx: __m512i,
    or: __m512i,
) -> __m512i {
    let live = _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(table.len())));
    // SAFETY: only lanes with `idx < min(len, 2^31)` (unsigned) are enabled;
    // each reads 4 bytes of `table`. Masked-off lanes read nothing and keep `or`.
    unsafe { _mm512_mask_i32gather_epi32::<4>(or, live, idx, table.as_ptr().cast()) }
}

#[archmage::arcane(import_intrinsics)]
fn gather_or_ps(_token: X64V4Token, table: &[f32], idx: __m512i, or: __m512) -> __m512 {
    let live = _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(table.len())));
    // SAFETY: only lanes with `idx < min(len, 2^31)` (unsigned) are enabled;
    // each reads 4 bytes of `table`. Masked-off lanes read nothing and keep `or`.
    unsafe { _mm512_mask_i32gather_ps::<4>(or, live, idx, table.as_ptr()) }
}

#[archmage::arcane(import_intrinsics)]
fn scatter_select_epi32<E: Int32>(
    _token: X64V4Token,
    dst: &mut [E],
    enable: u16,
    idx: __m512i,
    v: __m512i,
) {
    let live = enable & _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(dst.len())));
    // SAFETY: only enabled lanes with `idx < min(len, 2^31)` (unsigned) write,
    // each 4 bytes inside `dst`, which this call borrows exclusively. Any bit
    // pattern is a valid `E`.
    unsafe { _mm512_mask_i32scatter_epi32::<4>(dst.as_mut_ptr().cast(), live, idx, v) }
}

#[archmage::arcane(import_intrinsics)]
fn scatter_select_ps(_token: X64V4Token, dst: &mut [f32], enable: u16, idx: __m512i, v: __m512) {
    let live = enable & _mm512_cmplt_epu32_mask(idx, _mm512_set1_epi32(lane_bound(dst.len())));
    // SAFETY: only enabled lanes with `idx < min(len, 2^31)` (unsigned) write,
    // each 4 bytes inside `dst`, which this call borrows exclusively. Any bit
    // pattern is a valid `f32`.
    unsafe { _mm512_mask_i32scatter_ps::<4>(dst.as_mut_ptr(), live, idx, v) }
}

macro_rules! gather_scatter_methods {
    ($vec:ident, $elem:ty, $wrapping:ident, $or:ident, $scatter:ident) => {
        impl $vec<X64V4Token> {
            /// Loads `table[idx[i] & (N - 1)]` into lane `i`.
            ///
            /// `N` must be a power of two no larger than 2^31; other lengths
            /// fail to compile.
            #[inline(always)]
            pub fn gather_wrapping<const N: usize>(
                table: &[$elem; N],
                idx: u32x16<X64V4Token>,
            ) -> Self {
                Self($wrapping(idx.1, table, idx.0), idx.1)
            }

            /// Loads `table[idx[i]]` into lane `i`, or `or[i]` when `idx[i]` is
            /// out of range.
            ///
            /// Indices are unsigned. Indices at or above 2^31 count as out of
            /// range even when `table` is longer. Never panics; out-of-range
            /// lanes read no memory.
            #[inline(always)]
            pub fn gather_or(table: &[$elem], idx: u32x16<X64V4Token>, or: Self) -> Self {
                Self($or(idx.1, table, idx.0, or.0), idx.1)
            }

            /// Writes lane `i` to `dst[idx[i]]` when bit `i` of `enable` is set
            /// and `idx[i]` is in range; other lanes write nothing.
            ///
            /// Indices are unsigned, and indices at or above 2^31 count as out
            /// of range. Lanes are written in order, so when lanes share an
            /// index the highest one wins. Never panics.
            #[inline(always)]
            pub fn scatter_select(self, dst: &mut [$elem], enable: u16, idx: u32x16<X64V4Token>) {
                $scatter(self.1, dst, enable, idx.0, self.0)
            }
        }
    };
}

gather_scatter_methods!(
    u32x16,
    u32,
    gather_wrapping_epi32,
    gather_or_epi32,
    scatter_select_epi32
);
gather_scatter_methods!(
    i32x16,
    i32,
    gather_wrapping_epi32,
    gather_or_epi32,
    scatter_select_epi32
);
gather_scatter_methods!(
    f32x16,
    f32,
    gather_wrapping_ps,
    gather_or_ps,
    scatter_select_ps
);

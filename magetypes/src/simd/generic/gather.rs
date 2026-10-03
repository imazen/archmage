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
//! This file is safe code. The pointer-taking intrinsic calls, and the argument
//! that every access stays inside the borrow, live in `simd_storage.rs`
//! (`simd_storage::gather`) with the rest of magetypes' hand-written `unsafe`.

#![forbid(unsafe_code)]

use super::{f32x16, i32x16, u32x16};
use crate::simd_storage::gather as raw;
use archmage::X64V4Token;

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
                Self(raw::$wrapping(idx.1, table, idx.0), idx.1)
            }

            /// Loads `table[idx[i]]` into lane `i`, or `or[i]` when `idx[i]` is
            /// out of range.
            ///
            /// Indices are unsigned. Indices at or above 2^31 count as out of
            /// range even when `table` is longer. Never panics; out-of-range
            /// lanes read no memory.
            #[inline(always)]
            pub fn gather_or(table: &[$elem], idx: u32x16<X64V4Token>, or: Self) -> Self {
                Self(raw::$or(idx.1, table, idx.0, or.0), idx.1)
            }

            /// Writes lane `i` to `dst[idx[i]]` when bit `i` of `enable` is set
            /// and `idx[i]` is in range; other lanes write nothing.
            ///
            /// Indices are unsigned, and indices at or above 2^31 count as out
            /// of range. Lanes are written in order, so when lanes share an
            /// index the highest one wins. Never panics.
            #[inline(always)]
            pub fn scatter_select(self, dst: &mut [$elem], enable: u16, idx: u32x16<X64V4Token>) {
                raw::$scatter(self.1, dst, enable, idx.0, self.0)
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

//! Hand-written width-doubling and width-quadrupling polyfill backends.
//!
//! Each polyfilled backend delegates lane-wise to a narrower native backend on
//! the **same token** — zero `unsafe`, zero target intrinsics, and zero
//! `#[attune]` boundaries (the narrower native methods already establish the
//! feature boundary).
//!
//! Layout compatibility:
//! - 2-way (`X64V3Token` W512 over W256, `NeonToken`/`Wasm128Token` W256 over W128):
//!   `Repr = [HalfRepr; 2]`
//! - 4-way (`NeonToken` / `Wasm128Token` W512 over W128): `Repr = [QuarterRepr; 4]`
//!
//! Explicit array literals (`sub_array!`) are used instead of
//! `core::array::from_fn` so every polyfilled method avoids introducing a
//! distinct closure type and monomorphized `from_fn` symbol.

use crate::simd::backends::*;

// ============================================================================
// Unrolled 2-way / 4-way sub-vector helpers (zero closures)
// ============================================================================

/// Build a 2- or 4-element array by evaluating `$body` for each sub-vector index.
macro_rules! sub_array {
    (2, |$i:ident| $body:expr) => {
        [
            {
                let $i: usize = 0;
                $body
            },
            {
                let $i: usize = 1;
                $body
            },
        ]
    };
    (4, |$i:ident| $body:expr) => {
        [
            {
                let $i: usize = 0;
                $body
            },
            {
                let $i: usize = 1;
                $body
            },
            {
                let $i: usize = 2;
                $body
            },
            {
                let $i: usize = 3;
                $body
            },
        ]
    };
}

/// Evaluate `$body` for each sub-vector index `0..2` or `0..4`.
macro_rules! sub_for {
    (2, |$i:ident| $body:expr) => {{
        {
            let $i: usize = 0;
            $body;
        }
        {
            let $i: usize = 1;
            $body;
        }
    }};
    (4, |$i:ident| $body:expr) => {{
        {
            let $i: usize = 0;
            $body;
        }
        {
            let $i: usize = 1;
            $body;
        }
        {
            let $i: usize = 2;
            $body;
        }
        {
            let $i: usize = 3;
            $body;
        }
    }};
}

/// Reduce 2 or 4 sub-vectors into one using a vector `$combine` op, then run
/// `$reduce` once on the combined sub-vector (1 horizontal reduction instead of
/// 2 or 4, branchless for `all_true`/`any_true`).
///
/// For float reductions (`reduce_add`, `reduce_min`, `reduce_max`), lane-wise
/// folding matches native wider hardware (e.g., AVX-512 folding upper half into
/// lower half) and inherits the narrow backend's `min`/`max` NaN behavior
/// consistently across widths (e.g., NEON `f32x8`/`f32x16` match NEON `f32x4`
/// number-preferring `vminnmvq_f32`). Callers needing a specific NaN contract
/// use `reduce_min_with::<PROPAGATE/DROP_NAN>()` / `reduce_max_with::<...>()`.
macro_rules! sub_reduce_tree {
    (2, $sub:ident, $combine:ident, $reduce:ident, $self:expr, $a:expr) => {{
        let combined = <Self as $sub>::$combine($self, $a[0], $a[1]);
        <Self as $sub>::$reduce($self, combined)
    }};
    (4, $sub:ident, $combine:ident, $reduce:ident, $self:expr, $a:expr) => {{
        let m01 = <Self as $sub>::$combine($self, $a[0], $a[1]);
        let m23 = <Self as $sub>::$combine($self, $a[2], $a[3]);
        let combined = <Self as $sub>::$combine($self, m01, m23);
        <Self as $sub>::$reduce($self, combined)
    }};
}

/// Sum scalar `u32` reductions across 2 or 4 sub-vectors.
macro_rules! sub_sum_u32 {
    (2, |$i:ident| $body:expr) => {{
        ({
            let $i: usize = 0;
            $body
        }) + ({
            let $i: usize = 1;
            $body
        })
    }};
    (4, |$i:ident| $body:expr) => {{
        ({
            let $i: usize = 0;
            $body
        }) + ({
            let $i: usize = 1;
            $body
        }) + ({
            let $i: usize = 2;
            $body
        }) + ({
            let $i: usize = 3;
            $body
        })
    }};
}

/// Assemble a wide bitmask from 2 or 4 sub-vector bitmasks.
macro_rules! sub_bitmask {
    (2, $sub:ident, $sub_lanes:literal, $mask:ty, $self:expr, $a:expr) => {{
        let lo = <$mask>::from(<Self as $sub>::bitmask($self, $a[0]));
        let hi = <$mask>::from(<Self as $sub>::bitmask($self, $a[1]));
        lo | (hi << $sub_lanes)
    }};
    (4, $sub:ident, $sub_lanes:literal, $mask:ty, $self:expr, $a:expr) => {{
        let q0 = <$mask>::from(<Self as $sub>::bitmask($self, $a[0]));
        let q1 = <$mask>::from(<Self as $sub>::bitmask($self, $a[1]));
        let q2 = <$mask>::from(<Self as $sub>::bitmask($self, $a[2]));
        let q3 = <$mask>::from(<Self as $sub>::bitmask($self, $a[3]));
        q0 | (q1 << $sub_lanes) | (q2 << ($sub_lanes * 2)) | (q3 << ($sub_lanes * 3))
    }};
}

// ============================================================================
// Float backend polyfill (F32x8/F32x16, F64x4/F64x8)
// ============================================================================

macro_rules! impl_float_polyfill {
    (
        $token:ty, $wide:ident, $sub:ident,
        elem: $elem:ty, zero: $zero:expr,
        lanes: $lanes:literal, sub_lanes: $sub_lanes:literal, parts: $parts:tt
        $(, { $($extra:item)* })?
    ) => {
        impl $wide for $token {
            type Repr = [<Self as $sub>::Repr; $parts];

            #[inline(always)]
            fn splat(self, v: $elem) -> Self::Repr {
                [<Self as $sub>::splat(self, v); $parts]
            }

            #[inline(always)]
            fn zero(self) -> Self::Repr {
                [<Self as $sub>::zero(self); $parts]
            }

            #[inline(always)]
            fn load(self, data: &[$elem; $lanes]) -> Self::Repr {
                let (chunks, _) = data.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub>::load(self, &chunks[i]))
            }

            #[inline(always)]
            fn from_array(self, arr: [$elem; $lanes]) -> Self::Repr {
                let (chunks, _) = arr.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub>::from_array(self, chunks[i]))
            }

            #[inline(always)]
            fn store(self, repr: Self::Repr, out: &mut [$elem; $lanes]) {
                let (chunks, _) = out.as_chunks_mut::<$sub_lanes>();
                sub_for!($parts, |i| <Self as $sub>::store(
                    self,
                    repr[i],
                    &mut chunks[i]
                ));
            }

            #[inline(always)]
            fn to_array(self, repr: Self::Repr) -> [$elem; $lanes] {
                let mut out = [$zero; $lanes];
                let (chunks, _) = out.as_chunks_mut::<$sub_lanes>();
                sub_for!($parts, |i| chunks[i] =
                    <Self as $sub>::to_array(self, repr[i]));
                out
            }

            #[inline(always)]
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::add(self, a[i], b[i]))
            }

            #[inline(always)]
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::sub(self, a[i], b[i]))
            }

            #[inline(always)]
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul(self, a[i], b[i]))
            }

            #[inline(always)]
            fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::div(self, a[i], b[i]))
            }

            #[inline(always)]
            fn neg(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::neg(self, a[i]))
            }

            #[inline(always)]
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::min(self, a[i], b[i]))
            }

            #[inline(always)]
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::max(self, a[i], b[i]))
            }

            #[inline(always)]
            fn minimum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::minimum(self, a[i], b[i]))
            }

            #[inline(always)]
            fn maximum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::maximum(self, a[i], b[i]))
            }

            #[inline(always)]
            fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::min_fast(self, a[i], b[i]))
            }

            #[inline(always)]
            fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::max_fast(self, a[i], b[i]))
            }

            #[inline(always)]
            fn sqrt(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::sqrt(self, a[i]))
            }

            #[inline(always)]
            fn abs(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::abs(self, a[i]))
            }

            #[inline(always)]
            fn trunc(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::trunc(self, a[i]))
            }

            #[inline(always)]
            fn floor(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::floor(self, a[i]))
            }

            #[inline(always)]
            fn ceil(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::ceil(self, a[i]))
            }

            #[inline(always)]
            fn round(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::round(self, a[i]))
            }

            #[inline(always)]
            fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_add(self, a[i], b[i], c[i]))
            }

            #[inline(always)]
            fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_sub(self, a[i], b[i], c[i]))
            }

            #[inline(always)]
            fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_add_portable(
                    self, a[i], b[i], c[i]
                ))
            }

            #[inline(always)]
            fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_sub_portable(
                    self, a[i], b[i], c[i]
                ))
            }

            #[inline(always)]
            fn mul_neg_add_relaxed(
                self,
                a: Self::Repr,
                b: Self::Repr,
                c: Self::Repr,
            ) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_neg_add_relaxed(
                    self, a[i], b[i], c[i]
                ))
            }

            #[inline(always)]
            fn mul_neg_add_fused(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::mul_neg_add_fused(
                    self, a[i], b[i], c[i]
                ))
            }

            #[inline(always)]
            fn rcp_approx(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::rcp_approx(self, a[i]))
            }

            #[inline(always)]
            fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::rsqrt_approx(self, a[i]))
            }

            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::recip(self, a[i]))
            }

            #[inline(always)]
            fn rsqrt(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::rsqrt(self, a[i]))
            }

            #[inline(always)]
            fn reduce_add(self, a: Self::Repr) -> $elem {
                sub_reduce_tree!($parts, $sub, add, reduce_add, self, a)
            }

            #[inline(always)]
            fn reduce_min(self, a: Self::Repr) -> $elem {
                sub_reduce_tree!($parts, $sub, min, reduce_min, self, a)
            }

            #[inline(always)]
            fn reduce_max(self, a: Self::Repr) -> $elem {
                sub_reduce_tree!($parts, $sub, max, reduce_max, self, a)
            }

            #[inline(always)]
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_eq(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_ne(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_lt(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_le(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_gt(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_ge(self, a[i], b[i]))
            }

            #[inline(always)]
            fn blend(
                self,
                mask: Self::Repr,
                if_true: Self::Repr,
                if_false: Self::Repr,
            ) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::blend(
                    self,
                    mask[i],
                    if_true[i],
                    if_false[i]
                ))
            }

            #[inline(always)]
            fn not(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::not(self, a[i]))
            }

            #[inline(always)]
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitand(self, a[i], b[i]))
            }

            #[inline(always)]
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitor(self, a[i], b[i]))
            }

            #[inline(always)]
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitxor(self, a[i], b[i]))
            }

            $($($extra)*)?
        }
    };
}

// ============================================================================
// Integer backend polyfill + focused capability helpers
// ============================================================================

macro_rules! polyfill_int_mul {
    ($sub:ident, $parts:tt) => {
        #[inline(always)]
        fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::mul(self, a[i], b[i]))
        }
    };
}

macro_rules! polyfill_int_signed {
    ($sub:ident, $parts:tt) => {
        #[inline(always)]
        fn neg(self, a: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::neg(self, a[i]))
        }

        #[inline(always)]
        fn abs(self, a: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::abs(self, a[i]))
        }

        #[inline(always)]
        fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shr_arithmetic_const::<N>(
                self, a[i]
            ))
        }
    };
}

/// W512 unsigned traits declare `neg` and `shr_arithmetic_const`; the narrower
/// unsigned backends omit both, so `neg` computes `sub(zero, a)` and
/// `shr_arithmetic_const` delegates to `shr_logical_const`.
#[cfg(feature = "w512")]
macro_rules! polyfill_int_unsigned_w512 {
    ($sub:ident, $parts:tt) => {
        #[inline(always)]
        fn neg(self, a: Self::Repr) -> Self::Repr {
            let z = <Self as $sub>::zero(self);
            sub_array!($parts, |i| <Self as $sub>::sub(self, z, a[i]))
        }

        #[inline(always)]
        fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shr_logical_const::<N>(
                self, a[i]
            ))
        }
    };
}

macro_rules! polyfill_uniform_shifts {
    ($sub:ident, $parts:tt, $shr_arith_delegate:ident) => {
        #[inline(always)]
        fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shl_uniform(self, a[i], count))
        }

        #[inline(always)]
        fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shr_logical_uniform(
                self, a[i], count
            ))
        }

        #[inline(always)]
        fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::$shr_arith_delegate(
                self, a[i], count
            ))
        }
    };
}

/// W256 unsigned backends (`U16x16Backend`, `U32x8Backend`) only declare
/// `shl_uniform` and `shr_logical_uniform` (no `shr_arithmetic_uniform`).
#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! polyfill_uniform_shifts_unsigned_narrow {
    ($sub:ident, $parts:tt) => {
        #[inline(always)]
        fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shl_uniform(self, a[i], count))
        }

        #[inline(always)]
        fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::shr_logical_uniform(
                self, a[i], count
            ))
        }
    };
}

macro_rules! polyfill_saturating {
    ($sub:ident, $parts:tt) => {
        #[inline(always)]
        fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::saturating_add(self, a[i], b[i]))
        }

        #[inline(always)]
        fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::saturating_sub(self, a[i], b[i]))
        }
    };
}

macro_rules! polyfill_widen {
    ($sub:ident, $dst:ident, 2, $lo:ident, $hi:ident) => {
        #[inline(always)]
        fn $lo(self, a: Self::Repr) -> <Self as $dst>::Repr {
            [
                <Self as $sub>::$lo(self, a[0]),
                <Self as $sub>::$hi(self, a[0]),
            ]
        }

        #[inline(always)]
        fn $hi(self, a: Self::Repr) -> <Self as $dst>::Repr {
            [
                <Self as $sub>::$lo(self, a[1]),
                <Self as $sub>::$hi(self, a[1]),
            ]
        }
    };
    ($sub:ident, $dst:ident, 4, $lo:ident, $hi:ident) => {
        #[inline(always)]
        fn $lo(self, a: Self::Repr) -> <Self as $dst>::Repr {
            [
                <Self as $sub>::$lo(self, a[0]),
                <Self as $sub>::$hi(self, a[0]),
                <Self as $sub>::$lo(self, a[1]),
                <Self as $sub>::$hi(self, a[1]),
            ]
        }

        #[inline(always)]
        fn $hi(self, a: Self::Repr) -> <Self as $dst>::Repr {
            [
                <Self as $sub>::$lo(self, a[2]),
                <Self as $sub>::$hi(self, a[2]),
                <Self as $sub>::$lo(self, a[3]),
                <Self as $sub>::$hi(self, a[3]),
            ]
        }
    };
}

macro_rules! polyfill_narrow {
    ($sub:ident, $sdst:ident, $udst:ident, 2, $sfn:ident, $ufn:ident) => {
        #[inline(always)]
        fn $sfn(self, a: Self::Repr, b: Self::Repr) -> <Self as $sdst>::Repr {
            [
                <Self as $sub>::$sfn(self, a[0], a[1]),
                <Self as $sub>::$sfn(self, b[0], b[1]),
            ]
        }

        #[inline(always)]
        fn $ufn(self, a: Self::Repr, b: Self::Repr) -> <Self as $udst>::Repr {
            [
                <Self as $sub>::$ufn(self, a[0], a[1]),
                <Self as $sub>::$ufn(self, b[0], b[1]),
            ]
        }
    };
    ($sub:ident, $sdst:ident, $udst:ident, 4, $sfn:ident, $ufn:ident) => {
        #[inline(always)]
        fn $sfn(self, a: Self::Repr, b: Self::Repr) -> <Self as $sdst>::Repr {
            [
                <Self as $sub>::$sfn(self, a[0], a[1]),
                <Self as $sub>::$sfn(self, a[2], a[3]),
                <Self as $sub>::$sfn(self, b[0], b[1]),
                <Self as $sub>::$sfn(self, b[2], b[3]),
            ]
        }

        #[inline(always)]
        fn $ufn(self, a: Self::Repr, b: Self::Repr) -> <Self as $udst>::Repr {
            [
                <Self as $sub>::$ufn(self, a[0], a[1]),
                <Self as $sub>::$ufn(self, a[2], a[3]),
                <Self as $sub>::$ufn(self, b[0], b[1]),
                <Self as $sub>::$ufn(self, b[2], b[3]),
            ]
        }
    };
}

macro_rules! polyfill_i16_ops {
    ($sub:ident, $u16_dst:ident, $i32_dst:ident, $parts:tt) => {
        #[inline(always)]
        fn abs_diff(self, a: Self::Repr, b: Self::Repr) -> <Self as $u16_dst>::Repr {
            sub_array!($parts, |i| <Self as $sub>::abs_diff(self, a[i], b[i]))
        }

        #[inline(always)]
        fn madd_adjacent(self, a: Self::Repr, b: Self::Repr) -> <Self as $i32_dst>::Repr {
            sub_array!($parts, |i| <Self as $sub>::madd_adjacent(self, a[i], b[i]))
        }
    };
}

macro_rules! polyfill_u8_ops {
    ($sub:ident, $u16_dst:ident, $parts:tt) => {
        #[inline(always)]
        fn abs_diff(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
            sub_array!($parts, |i| <Self as $sub>::abs_diff(self, a[i], b[i]))
        }

        #[inline(always)]
        fn reduce_add_u32(self, a: Self::Repr) -> u32 {
            sub_sum_u32!($parts, |i| <Self as $sub>::reduce_add_u32(self, a[i]))
        }

        #[inline(always)]
        fn sum_abs_diff(self, a: Self::Repr, b: Self::Repr) -> u32 {
            sub_sum_u32!($parts, |i| <Self as $sub>::sum_abs_diff(self, a[i], b[i]))
        }

        #[inline(always)]
        fn pairwise_widen_add(self, a: Self::Repr) -> <Self as $u16_dst>::Repr {
            sub_array!($parts, |i| <Self as $sub>::pairwise_widen_add(self, a[i]))
        }
    };
}

macro_rules! polyfill_pairwise_widen_add {
    ($sub:ident, $dst:ident, $parts:tt) => {
        #[inline(always)]
        fn pairwise_widen_add(self, a: Self::Repr) -> <Self as $dst>::Repr {
            sub_array!($parts, |i| <Self as $sub>::pairwise_widen_add(self, a[i]))
        }
    };
}

macro_rules! impl_int_polyfill {
    (
        $token:ty, $wide:ident, $sub:ident,
        elem: $elem:ty, lanes: $lanes:literal, sub_lanes: $sub_lanes:literal,
        parts: $parts:tt, mask: $mask:ty
        $(, { $($extra:item)* })?
    ) => {
        impl $wide for $token {
            type Repr = [<Self as $sub>::Repr; $parts];

            #[inline(always)]
            fn splat(self, v: $elem) -> Self::Repr {
                [<Self as $sub>::splat(self, v); $parts]
            }

            #[inline(always)]
            fn zero(self) -> Self::Repr {
                [<Self as $sub>::zero(self); $parts]
            }

            #[inline(always)]
            fn load(self, data: &[$elem; $lanes]) -> Self::Repr {
                let (chunks, _) = data.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub>::load(self, &chunks[i]))
            }

            #[inline(always)]
            fn from_array(self, arr: [$elem; $lanes]) -> Self::Repr {
                let (chunks, _) = arr.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub>::from_array(self, chunks[i]))
            }

            #[inline(always)]
            fn store(self, repr: Self::Repr, out: &mut [$elem; $lanes]) {
                let (chunks, _) = out.as_chunks_mut::<$sub_lanes>();
                sub_for!($parts, |i| <Self as $sub>::store(
                    self,
                    repr[i],
                    &mut chunks[i]
                ));
            }

            #[inline(always)]
            fn to_array(self, repr: Self::Repr) -> [$elem; $lanes] {
                let mut out = [0; $lanes];
                let (chunks, _) = out.as_chunks_mut::<$sub_lanes>();
                sub_for!($parts, |i| chunks[i] = <Self as $sub>::to_array(self, repr[i]));
                out
            }

            #[inline(always)]
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::add(self, a[i], b[i]))
            }

            #[inline(always)]
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::sub(self, a[i], b[i]))
            }

            #[inline(always)]
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::min(self, a[i], b[i]))
            }

            #[inline(always)]
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::max(self, a[i], b[i]))
            }

            #[inline(always)]
            fn reduce_add(self, a: Self::Repr) -> $elem {
                sub_reduce_tree!($parts, $sub, add, reduce_add, self, a)
            }

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::shl_const::<N>(self, a[i]))
            }

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::shr_logical_const::<N>(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn all_true(self, a: Self::Repr) -> bool {
                sub_reduce_tree!($parts, $sub, bitand, all_true, self, a)
            }

            #[inline(always)]
            fn any_true(self, a: Self::Repr) -> bool {
                sub_reduce_tree!($parts, $sub, bitor, any_true, self, a)
            }

            #[inline(always)]
            fn bitmask(self, a: Self::Repr) -> $mask {
                sub_bitmask!($parts, $sub, $sub_lanes, $mask, self, a)
            }

            #[inline(always)]
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_eq(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_ne(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_lt(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_le(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_gt(self, a[i], b[i]))
            }

            #[inline(always)]
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::simd_ge(self, a[i], b[i]))
            }

            #[inline(always)]
            fn blend(
                self,
                mask: Self::Repr,
                if_true: Self::Repr,
                if_false: Self::Repr,
            ) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::blend(
                    self,
                    mask[i],
                    if_true[i],
                    if_false[i]
                ))
            }

            #[inline(always)]
            fn not(self, a: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::not(self, a[i]))
            }

            #[inline(always)]
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitand(self, a[i], b[i]))
            }

            #[inline(always)]
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitor(self, a[i], b[i]))
            }

            #[inline(always)]
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {
                sub_array!($parts, |i| <Self as $sub>::bitxor(self, a[i], b[i]))
            }

            $($($extra)*)?
        }
    };
}

// ============================================================================
// F32 ↔ I32 conversion polyfill (F32x16Convert)
// ============================================================================

macro_rules! impl_f32_convert_polyfill {
    ($token:ty, $wide_conv:ident, $sub_conv:ident, $wide_f32:ident, $wide_i32:ident, parts: $parts:tt) => {
        impl $wide_conv for $token {
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: <Self as $wide_f32>::Repr) -> <Self as $wide_i32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::bitcast_f32_to_i32(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: <Self as $wide_i32>::Repr) -> <Self as $wide_f32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::bitcast_i32_to_f32(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn convert_f32_to_i32(self, a: <Self as $wide_f32>::Repr) -> <Self as $wide_i32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::convert_f32_to_i32(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn convert_f32_to_i32_saturating(
                self,
                a: <Self as $wide_f32>::Repr,
            ) -> <Self as $wide_i32>::Repr {
                sub_array!($parts, |i| {
                    <Self as $sub_conv>::convert_f32_to_i32_saturating(self, a[i])
                })
            }

            #[inline(always)]
            fn convert_f32_to_i32_fast(
                self,
                a: <Self as $wide_f32>::Repr,
            ) -> <Self as $wide_i32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::convert_f32_to_i32_fast(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn convert_f32_to_i32_round(
                self,
                a: <Self as $wide_f32>::Repr,
            ) -> <Self as $wide_i32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::convert_f32_to_i32_round(
                    self, a[i]
                ))
            }

            #[inline(always)]
            fn convert_i32_to_f32(self, a: <Self as $wide_i32>::Repr) -> <Self as $wide_f32>::Repr {
                sub_array!($parts, |i| <Self as $sub_conv>::convert_i32_to_f32(
                    self, a[i]
                ))
            }
        }
    };
}

// ============================================================================
// Pair bitcast polyfill (W256 on NEON / WASM128)
// ============================================================================

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! impl_bitcast_pair_polyfill {
    (
        $token:ty, $wide:ident, $sub:ident,
        $a_wide:ident, $b_wide:ident, $fn_ab:ident, $fn_ba:ident
    ) => {
        impl $wide for $token {
            #[inline(always)]
            fn $fn_ab(self, a: <Self as $a_wide>::Repr) -> <Self as $b_wide>::Repr {
                [
                    <Self as $sub>::$fn_ab(self, a[0]),
                    <Self as $sub>::$fn_ab(self, a[1]),
                ]
            }

            #[inline(always)]
            fn $fn_ba(self, a: <Self as $b_wide>::Repr) -> <Self as $a_wide>::Repr {
                [
                    <Self as $sub>::$fn_ba(self, a[0]),
                    <Self as $sub>::$fn_ba(self, a[1]),
                ]
            }
        }
    };
}

// ============================================================================
// W256 bundle (all 10 W256 types + F32x8Convert + 5 W256 Bitcast traits)
// ============================================================================

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
macro_rules! impl_w256_bundle {
    ($token:ty, to_u8_bytes: $to_u8_helper:path) => {
        impl_float_polyfill!(
            $token, F32x8Backend, F32x4Backend,
            elem: f32, zero: 0.0f32, lanes: 8, sub_lanes: 4, parts: 2,
            {
                #[inline(always)]
                fn to_u8_bytes(self, a: Self::Repr) -> [u8; 8] {
                    $to_u8_helper(self, a)
                }
            }
        );
        impl_float_polyfill!(
            $token, F64x4Backend, F64x2Backend,
            elem: f64, zero: 0.0f64, lanes: 4, sub_lanes: 2, parts: 2
        );
        impl_int_polyfill!(
            $token, I8x32Backend, I8x16Backend,
            elem: i8, lanes: 32, sub_lanes: 16, parts: 2, mask: u32,
            {
                polyfill_int_signed!(I8x16Backend, 2);
                polyfill_saturating!(I8x16Backend, 2);
                polyfill_widen!(I8x16Backend, I16x16Backend, 2, widen_low_i8_to_i16, widen_high_i8_to_i16);
            }
        );
        impl_int_polyfill!(
            $token, U8x32Backend, U8x16Backend,
            elem: u8, lanes: 32, sub_lanes: 16, parts: 2, mask: u32,
            {
                polyfill_saturating!(U8x16Backend, 2);
                polyfill_widen!(U8x16Backend, U16x16Backend, 2, widen_low_u8_to_u16, widen_high_u8_to_u16);
                polyfill_u8_ops!(U8x16Backend, U16x16Backend, 2);
            }
        );
        impl_int_polyfill!(
            $token, I16x16Backend, I16x8Backend,
            elem: i16, lanes: 16, sub_lanes: 8, parts: 2, mask: u32,
            {
                polyfill_int_mul!(I16x8Backend, 2);
                polyfill_int_signed!(I16x8Backend, 2);
                polyfill_uniform_shifts!(I16x8Backend, 2, shr_arithmetic_uniform);
                polyfill_saturating!(I16x8Backend, 2);
                polyfill_widen!(I16x8Backend, I32x8Backend, 2, widen_low_i16_to_i32, widen_high_i16_to_i32);
                polyfill_narrow!(I16x8Backend, I8x32Backend, U8x32Backend, 2, narrow_saturating_i16_to_i8, narrow_saturating_i16_to_u8);
                polyfill_i16_ops!(I16x8Backend, U16x16Backend, I32x8Backend, 2);
            }
        );
        impl_int_polyfill!(
            $token, U16x16Backend, U16x8Backend,
            elem: u16, lanes: 16, sub_lanes: 8, parts: 2, mask: u32,
            {
                polyfill_int_mul!(U16x8Backend, 2);
                polyfill_uniform_shifts_unsigned_narrow!(U16x8Backend, 2);
                polyfill_saturating!(U16x8Backend, 2);
                polyfill_widen!(U16x8Backend, U32x8Backend, 2, widen_low_u16_to_u32, widen_high_u16_to_u32);
                polyfill_pairwise_widen_add!(U16x8Backend, U32x8Backend, 2);
            }
        );
        impl_int_polyfill!(
            $token, I32x8Backend, I32x4Backend,
            elem: i32, lanes: 8, sub_lanes: 4, parts: 2, mask: u32,
            {
                polyfill_int_mul!(I32x4Backend, 2);
                polyfill_int_signed!(I32x4Backend, 2);
                polyfill_uniform_shifts!(I32x4Backend, 2, shr_arithmetic_uniform);
                polyfill_narrow!(I32x4Backend, I16x16Backend, U16x16Backend, 2, narrow_saturating_i32_to_i16, narrow_saturating_i32_to_u16);
            }
        );
        impl_int_polyfill!(
            $token, U32x8Backend, U32x4Backend,
            elem: u32, lanes: 8, sub_lanes: 4, parts: 2, mask: u32,
            {
                polyfill_int_mul!(U32x4Backend, 2);
                polyfill_uniform_shifts_unsigned_narrow!(U32x4Backend, 2);
            }
        );
        impl_int_polyfill!(
            $token, I64x4Backend, I64x2Backend,
            elem: i64, lanes: 4, sub_lanes: 2, parts: 2, mask: u32,
            {
                polyfill_int_signed!(I64x2Backend, 2);
            }
        );
        impl_int_polyfill!(
            $token, U64x4Backend, U64x2Backend,
            elem: u64, lanes: 4, sub_lanes: 2, parts: 2, mask: u32
        );
        impl_f32_convert_polyfill!(
            $token, F32x8Convert, F32x4Convert, F32x8Backend, I32x8Backend, parts: 2
        );
        impl_bitcast_pair_polyfill!(
            $token, U32x8Bitcast, U32x4Bitcast, U32x8Backend, I32x8Backend,
            bitcast_u32_to_i32, bitcast_i32_to_u32
        );
        impl_bitcast_pair_polyfill!(
            $token, I64x4Bitcast, I64x2Bitcast, I64x4Backend, F64x4Backend,
            bitcast_i64_to_f64, bitcast_f64_to_i64
        );
        impl_bitcast_pair_polyfill!(
            $token, I8x32Bitcast, I8x16Bitcast, I8x32Backend, U8x32Backend,
            bitcast_i8_to_u8, bitcast_u8_to_i8
        );
        impl_bitcast_pair_polyfill!(
            $token, I16x16Bitcast, I16x8Bitcast, I16x16Backend, U16x16Backend,
            bitcast_i16_to_u16, bitcast_u16_to_i16
        );
        impl_bitcast_pair_polyfill!(
            $token, U64x4Bitcast, U64x2Bitcast, U64x4Backend, I64x4Backend,
            bitcast_u64_to_i64, bitcast_i64_to_u64
        );
    };
}

// ============================================================================
// F32x16Internal layout/conversion polyfill
// ============================================================================

#[cfg(feature = "w512")]
macro_rules! polyfill_rgba_wide_ops {
    (2) => {
        #[inline(always)]
        fn deinterleave_rgba_wide(self, rgba: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4] {
            use crate::simd::generic::f32x8 as V;
            let g0 = V::deinterleave_4ch([
                V::from_repr_unchecked(self, rgba[0][0]),
                V::from_repr_unchecked(self, rgba[0][1]),
                V::from_repr_unchecked(self, rgba[1][0]),
                V::from_repr_unchecked(self, rgba[1][1]),
            ]);
            let g1 = V::deinterleave_4ch([
                V::from_repr_unchecked(self, rgba[2][0]),
                V::from_repr_unchecked(self, rgba[2][1]),
                V::from_repr_unchecked(self, rgba[3][0]),
                V::from_repr_unchecked(self, rgba[3][1]),
            ]);
            [
                [g0[0].into_repr(), g1[0].into_repr()],
                [g0[1].into_repr(), g1[1].into_repr()],
                [g0[2].into_repr(), g1[2].into_repr()],
                [g0[3].into_repr(), g1[3].into_repr()],
            ]
        }

        #[inline(always)]
        fn interleave_rgba_wide(self, ch: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4] {
            use crate::simd::generic::f32x8 as V;
            let g0 = V::interleave_4ch([
                V::from_repr_unchecked(self, ch[0][0]),
                V::from_repr_unchecked(self, ch[1][0]),
                V::from_repr_unchecked(self, ch[2][0]),
                V::from_repr_unchecked(self, ch[3][0]),
            ]);
            let g1 = V::interleave_4ch([
                V::from_repr_unchecked(self, ch[0][1]),
                V::from_repr_unchecked(self, ch[1][1]),
                V::from_repr_unchecked(self, ch[2][1]),
                V::from_repr_unchecked(self, ch[3][1]),
            ]);
            [
                [g0[0].into_repr(), g0[1].into_repr()],
                [g0[2].into_repr(), g0[3].into_repr()],
                [g1[0].into_repr(), g1[1].into_repr()],
                [g1[2].into_repr(), g1[3].into_repr()],
            ]
        }
    };
    (4) => {
        #[inline(always)]
        fn deinterleave_rgba_wide(self, rgba: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4] {
            use crate::simd::generic::f32x4 as V;
            let g = sub_array!(4, |i| V::deinterleave_4ch(sub_array!(4, |j| {
                V::from_repr_unchecked(self, rgba[i][j])
            })));
            sub_array!(4, |c| sub_array!(4, |i| g[i][c].into_repr()))
        }

        #[inline(always)]
        fn interleave_rgba_wide(self, ch: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4] {
            use crate::simd::generic::f32x4 as V;
            let g = sub_array!(4, |i| V::interleave_4ch(sub_array!(4, |c| {
                V::from_repr_unchecked(self, ch[c][i])
            })));
            sub_array!(4, |i| sub_array!(4, |j| g[i][j].into_repr()))
        }
    };
}

#[cfg(feature = "w512")]
macro_rules! impl_f32x16_internal_polyfill {
    (
        $token:ty, $sub_int:ident, $sub_vec:ident,
        sub_lanes: $sub_lanes:literal, parts: $parts:tt,
        to_u16: |$s:ident, $v:ident, $cfg:ident| $to_u16:expr
    ) => {
        impl F32x16Internal for $token {
            type InternalRepr = [<Self as $sub_int>::InternalRepr; $parts];

            #[inline(always)]
            fn from_u16_lanes(self, lanes: &[u16; 16]) -> Self::InternalRepr {
                let (chunks, _) = lanes.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub_int>::from_u16_lanes(
                    self, &chunks[i]
                ))
            }

            #[inline(always)]
            fn to_u16_lanes<const $cfg: u32>(self, $v: Self::InternalRepr) -> [u16; 16] {
                let $s = self;
                $to_u16
            }

            #[inline(always)]
            fn to_u8_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u8; 16] {
                crate::simd_storage::cast(sub_array!($parts, |i| {
                    <Self as $sub_int>::to_u8_lanes::<CFG>(self, value[i])
                }))
            }

            #[inline(always)]
            fn to_u32_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> Self::InternalRepr {
                sub_array!($parts, |i| <Self as $sub_int>::to_u32_lanes::<CFG>(
                    self, value[i]
                ))
            }

            #[inline(always)]
            fn from_u32_lanes(self, value: Self::InternalRepr) -> Self::InternalRepr {
                sub_array!($parts, |i| <Self as $sub_int>::from_u32_lanes(
                    self, value[i]
                ))
            }

            #[inline(always)]
            fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 16]) -> Self::InternalRepr {
                let (chunks, _) = lanes.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub_int>::from_f16_lanes::<CFG>(
                    self, &chunks[i]
                ))
            }

            #[inline(always)]
            fn to_f16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; 16] {
                crate::simd_storage::cast(sub_array!($parts, |i| {
                    <Self as $sub_int>::to_f16_lanes::<CFG>(self, value[i])
                }))
            }

            #[inline(always)]
            fn from_bf16_lanes(self, lanes: &[u16; 16]) -> Self::InternalRepr {
                let (chunks, _) = lanes.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| <Self as $sub_int>::from_bf16_lanes(
                    self, &chunks[i]
                ))
            }

            #[inline(always)]
            fn to_bf16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; 16] {
                crate::simd_storage::cast(sub_array!($parts, |i| {
                    <Self as $sub_int>::to_bf16_lanes::<CFG>(self, value[i])
                }))
            }

            #[inline(always)]
            fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> Self::InternalRepr {
                let (chunks, _) = lanes.as_chunks::<$sub_lanes>();
                sub_array!($parts, |i| {
                    crate::simd::generic::$sub_vec::from_u8_t(self, &chunks[i]).into_repr()
                })
            }

            polyfill_rgba_wide_ops!($parts);

            #[inline(always)]
            fn broadcast_rgba_alpha_repr(self, value: Self::InternalRepr) -> Self::InternalRepr {
                sub_array!($parts, |i| <Self as $sub_int>::broadcast_rgba_alpha_repr(
                    self, value[i]
                ))
            }

            #[inline(always)]
            fn with_rgba_alpha_repr(
                self,
                value: Self::InternalRepr,
                src: Self::InternalRepr,
            ) -> Self::InternalRepr {
                sub_array!($parts, |i| <Self as $sub_int>::with_rgba_alpha_repr(
                    self, value[i], src[i]
                ))
            }
        }
    };
}

// ============================================================================
// W512 bundle (all 10 W512 types + F32x16Convert)
// ============================================================================

#[cfg(feature = "w512")]
macro_rules! impl_w512_bundle {
    (
        $token:ty, parts: $parts:tt,
        f32: ($f32_sub:ident, $f32_sub_lanes:literal),
        f64: ($f64_sub:ident, $f64_sub_lanes:literal),
        i8:  ($i8_sub:ident,  $i8_sub_lanes:literal),
        u8:  ($u8_sub:ident,  $u8_sub_lanes:literal),
        i16: ($i16_sub:ident, $i16_sub_lanes:literal),
        u16: ($u16_sub:ident, $u16_sub_lanes:literal),
        i32: ($i32_sub:ident, $i32_sub_lanes:literal),
        u32: ($u32_sub:ident, $u32_sub_lanes:literal),
        i64: ($i64_sub:ident, $i64_sub_lanes:literal),
        u64: ($u64_sub:ident, $u64_sub_lanes:literal),
        f32_conv: $f32_conv_sub:ident
    ) => {
        impl_float_polyfill!(
            $token, F32x16Backend, $f32_sub,
            elem: f32, zero: 0.0f32, lanes: 16, sub_lanes: $f32_sub_lanes, parts: $parts
        );
        impl_float_polyfill!(
            $token, F64x8Backend, $f64_sub,
            elem: f64, zero: 0.0f64, lanes: 8, sub_lanes: $f64_sub_lanes, parts: $parts
        );
        impl_int_polyfill!(
            $token, I8x64Backend, $i8_sub,
            elem: i8, lanes: 64, sub_lanes: $i8_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_signed!($i8_sub, $parts);
                polyfill_saturating!($i8_sub, $parts);
                polyfill_widen!($i8_sub, I16x32Backend, $parts, widen_low_i8_to_i16, widen_high_i8_to_i16);
            }
        );
        impl_int_polyfill!(
            $token, U8x64Backend, $u8_sub,
            elem: u8, lanes: 64, sub_lanes: $u8_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_unsigned_w512!($u8_sub, $parts);
                polyfill_saturating!($u8_sub, $parts);
                polyfill_widen!($u8_sub, U16x32Backend, $parts, widen_low_u8_to_u16, widen_high_u8_to_u16);
                polyfill_u8_ops!($u8_sub, U16x32Backend, $parts);
            }
        );
        impl_int_polyfill!(
            $token, I16x32Backend, $i16_sub,
            elem: i16, lanes: 32, sub_lanes: $i16_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_mul!($i16_sub, $parts);
                polyfill_int_signed!($i16_sub, $parts);
                polyfill_uniform_shifts!($i16_sub, $parts, shr_arithmetic_uniform);
                polyfill_saturating!($i16_sub, $parts);
                polyfill_widen!($i16_sub, I32x16Backend, $parts, widen_low_i16_to_i32, widen_high_i16_to_i32);
                polyfill_narrow!($i16_sub, I8x64Backend, U8x64Backend, $parts, narrow_saturating_i16_to_i8, narrow_saturating_i16_to_u8);
                polyfill_i16_ops!($i16_sub, U16x32Backend, I32x16Backend, $parts);
            }
        );
        impl_int_polyfill!(
            $token, U16x32Backend, $u16_sub,
            elem: u16, lanes: 32, sub_lanes: $u16_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_mul!($u16_sub, $parts);
                polyfill_int_unsigned_w512!($u16_sub, $parts);
                polyfill_uniform_shifts!($u16_sub, $parts, shr_logical_uniform);
                polyfill_saturating!($u16_sub, $parts);
                polyfill_widen!($u16_sub, U32x16Backend, $parts, widen_low_u16_to_u32, widen_high_u16_to_u32);
                polyfill_pairwise_widen_add!($u16_sub, U32x16Backend, $parts);
            }
        );
        impl_int_polyfill!(
            $token, I32x16Backend, $i32_sub,
            elem: i32, lanes: 16, sub_lanes: $i32_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_mul!($i32_sub, $parts);
                polyfill_int_signed!($i32_sub, $parts);
                polyfill_uniform_shifts!($i32_sub, $parts, shr_arithmetic_uniform);
                polyfill_narrow!($i32_sub, I16x32Backend, U16x32Backend, $parts, narrow_saturating_i32_to_i16, narrow_saturating_i32_to_u16);
            }
        );
        impl_int_polyfill!(
            $token, U32x16Backend, $u32_sub,
            elem: u32, lanes: 16, sub_lanes: $u32_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_mul!($u32_sub, $parts);
                polyfill_int_unsigned_w512!($u32_sub, $parts);
                polyfill_uniform_shifts!($u32_sub, $parts, shr_logical_uniform);
            }
        );
        impl_int_polyfill!(
            $token, I64x8Backend, $i64_sub,
            elem: i64, lanes: 8, sub_lanes: $i64_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_signed!($i64_sub, $parts);
            }
        );
        impl_int_polyfill!(
            $token, U64x8Backend, $u64_sub,
            elem: u64, lanes: 8, sub_lanes: $u64_sub_lanes, parts: $parts, mask: u64,
            {
                polyfill_int_unsigned_w512!($u64_sub, $parts);
            }
        );
        impl_f32_convert_polyfill!(
            $token, F32x16Convert, $f32_conv_sub, F32x16Backend, I32x16Backend, parts: $parts
        );
    };
}

// ============================================================================
// Instantiations
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_w256 {
    use super::*;
    use archmage::NeonToken;

    impl_w256_bundle!(NeonToken, to_u8_bytes: super::super::arm_neon::f32x8_to_u8_bytes);
}

#[cfg(target_arch = "wasm32")]
mod wasm128_w256 {
    use super::*;
    use archmage::Wasm128Token;

    impl_w256_bundle!(Wasm128Token, to_u8_bytes: super::super::wasm128::f32x8_to_u8_bytes);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
mod x86_v3_w512 {
    use super::*;
    use archmage::X64V3Token;

    impl_w512_bundle!(
        X64V3Token, parts: 2,
        f32: (F32x8Backend, 8),
        f64: (F64x4Backend, 4),
        i8:  (I8x32Backend, 32),
        u8:  (U8x32Backend, 32),
        i16: (I16x16Backend, 16),
        u16: (U16x16Backend, 16),
        i32: (I32x8Backend, 8),
        u32: (U32x8Backend, 8),
        i64: (I64x4Backend, 4),
        u64: (U64x4Backend, 4),
        f32_conv: F32x8Convert
    );

    impl_f32x16_internal_polyfill!(
        X64V3Token, F32x8Internal, f32x8, sub_lanes: 8, parts: 2,
        to_u16: |s, v, CFG| super::super::x86_v3::f32x16_to_u16_lanes::<CFG>(s, v)
    );
}

#[cfg(all(target_arch = "aarch64", feature = "w512"))]
mod neon_w512 {
    use super::*;
    use archmage::NeonToken;

    impl_w512_bundle!(
        NeonToken, parts: 4,
        f32: (F32x4Backend, 4),
        f64: (F64x2Backend, 2),
        i8:  (I8x16Backend, 16),
        u8:  (U8x16Backend, 16),
        i16: (I16x8Backend, 8),
        u16: (U16x8Backend, 8),
        i32: (I32x4Backend, 4),
        u32: (U32x4Backend, 4),
        i64: (I64x2Backend, 2),
        u64: (U64x2Backend, 2),
        f32_conv: F32x4Convert
    );

    impl_f32x16_internal_polyfill!(
        NeonToken, F32x4Internal, f32x4, sub_lanes: 4, parts: 4,
        to_u16: |s, v, CFG| crate::simd_storage::cast(sub_array!(4, |i| {
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(s, v[i])
        }))
    );
}

#[cfg(all(target_arch = "wasm32", feature = "w512"))]
mod wasm128_w512 {
    use super::*;
    use archmage::Wasm128Token;

    impl_w512_bundle!(
        Wasm128Token, parts: 4,
        f32: (F32x4Backend, 4),
        f64: (F64x2Backend, 2),
        i8:  (I8x16Backend, 16),
        u8:  (U8x16Backend, 16),
        i16: (I16x8Backend, 8),
        u16: (U16x8Backend, 8),
        i32: (I32x4Backend, 4),
        u32: (U32x4Backend, 4),
        i64: (I64x2Backend, 2),
        u64: (U64x2Backend, 2),
        f32_conv: F32x4Convert
    );

    impl_f32x16_internal_polyfill!(
        Wasm128Token, F32x4Internal, f32x4, sub_lanes: 4, parts: 4,
        to_u16: |s, v, CFG| crate::simd_storage::cast(sub_array!(4, |i| {
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(s, v[i])
        }))
    );
}

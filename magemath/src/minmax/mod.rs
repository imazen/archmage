//! Const-configured f32/f64 min/max, with independent NaN, zero and input policies.
//!
//! Number prefers a non-NaN over either quiet or signaling NaN; Propagate returns
//! NaN if either operand is NaN. Ordered zeros choose -0 for min and +0 for max.
//! NaN payload/sign/signaling encoding and exception flags are unspecified.
//! Normal-domain graphs need no special-value repairs. AssumeValid removes the
//! input checks too; violations never cause memory unsoundness.
//!
//! # Zero clamps: use Full or Finite
//!
//! `NORMAL` and `UNCHECKED_NORMAL` are the specialized **finite normal operands
//! only, zero excluded** presets. Both operands must satisfy that restriction:
//! `min(x, 0)` and `max(0, 5)` are outside it. Checked Normal rejects them;
//! unchecked Normal has unspecified output. `NONZERO` alone would still omit
//! the exclusion of subnormals and infinities.
//!
//! Finite admits zeros and subnormals. Keep Ordered zeros for portable zero
//! signs, or explicitly select Backend zero ties when their signs do not matter.
//! Full-domain presets also admit zeros, with their stated NaN/zero semantics.
//!
//! ```
//! use magemath::minmax::{
//!     MinMaxDomain, MinMaxOptions, MinMaxValidation, try_min_with, try_max_with,
//! };
//! const FINITE: u32 = MinMaxOptions::new()
//!     .with_domain(MinMaxDomain::Finite)
//!     .with_validation(MinMaxValidation::AssumeValid)
//!     .bits();
//! assert_eq!(try_min_with::<FINITE>(0.25, 0.0).unwrap(), 0.0);
//! assert_eq!(try_max_with::<FINITE>(0.0, 5.0).unwrap(), 5.0);
//! assert!(try_max_with::<{ MinMaxOptions::NORMAL.bits() }>(0.0, 5.0).is_err());
//! ```
//!
//! Typed vector and width-specific lane entries retain their caller's token.
//! Scalar/array entries select once unless configured for Caller execution.
//!
//! # Presets and policy permutations
//!
//! | [`crate::minmax::MinMaxOptions`] preset | Domain | NaN policy | Opposite-zero tie | Validation |
//! |---|---|---|---|---|
//! | `PORTABLE_NUMBER` (also `new()`) | Full | Prefer the number; two NaNs yield NaN | Min -0, max +0 | Checked; no domain rejection needed |
//! | `PORTABLE_PROPAGATE` | Full | Either NaN yields NaN | Min -0, max +0 | Checked; no domain rejection needed |
//! | `RELAXED` | Full | Backend-dependent | Backend-dependent | Checked; no domain rejection needed |
//! | `NORMAL` | Normal | No NaNs admitted | No zeros admitted | Checked |
//! | `UNCHECKED_NORMAL` | Normal | No NaNs admitted | No zeros admitted | AssumeValid |
//!
//! All presets default to Automatic execution. Builders can independently select
//! any [`crate::minmax::MinMaxNaN`], [`crate::minmax::MinMaxZero`],
//! [`crate::minmax::MinMaxDomain`], [`crate::minmax::MinMaxValidation`] and
//! [`crate::minmax::MinMaxExecution`] combination, then encode it with `.bits()`.
//!
//! | Domain | Accepted inputs | Distinct result policies |
//! |---|---|---|
//! | Full | Every floating-point value | All three NaN × both zero choices |
//! | Finite | Includes signed zeros and subnormals; excludes infinities/NaNs | Both zero choices; NaN choices encode identically |
//! | Normal | Finite normal values; excludes zeros/subnormals/infinities/NaNs | NaN and zero choices all encode identically |
//!
//! Full-domain validation choices encode identically. On Finite/Normal,
//! Checked rejects the whole vector with an invalid-lane mask; AssumeValid
//! removes validation. Automatic / Caller execution is separate from result
//! policy: a portable result may use one hardware instruction on a suitable
//! ISA. Relaxed behavior permits ISA-dependent results; it is not a speed claim.
//! Min/max has no precision or FMA selector because neither describes this
//! operation's numerical choices.
#![doc = include_str!("permutations.md")]
// BEGIN GENERATED TIMING DOCUMENTS
#![doc = include_str!("timings/mac-neon-min.md")]
#![doc = include_str!("timings/mac-neon-max.md")]
#![doc = include_str!("timings/i265-v3-min.md")]
#![doc = include_str!("timings/i265-v3-max.md")]
#![doc = include_str!("timings/dev-scalar-min.md")]
#![doc = include_str!("timings/dev-scalar-max.md")]
#![doc = include_str!("timings/dev-v3-min.md")]
#![doc = include_str!("timings/dev-v3-max.md")]
#![doc = include_str!("timings/dev-v4-min.md")]
#![doc = include_str!("timings/dev-v4-max.md")]
#![doc = include_str!("timings/dev-v4x-min.md")]
#![doc = include_str!("timings/dev-v4x-max.md")]
#![doc = include_str!("timings/wasm-strict-min.md")]
#![doc = include_str!("timings/wasm-strict-max.md")]
// END GENERATED TIMING DOCUMENTS
mod api;
mod options;
use crate::{PrimitiveError, PrimitiveErrorKind};
pub use api::*;
pub use options::*;

macro_rules! graph {
    ($name:ident, $v:ident, $bound:ident, $float:ident, $n:literal) => {
        #[inline(always)]
        pub(super) fn $name<const O: u32, const MAX: bool, T: magetypes::simd::backends::$bound>(
            t: T,
            a: magetypes::simd::generic::$v<T>,
            b: magetypes::simd::generic::$v<T>,
        ) -> Result<magetypes::simd::generic::$v<T>, PrimitiveError> {
            const {
                assert!(MinMaxOptions::valid(O), "invalid minmax options");
            }
            use magetypes::simd::generic::$v as V;
            let domain = O & 3;
            if domain != 0 && O & MinMaxOptions::ASSUME == 0 {
                let lo = V::splat_t(
                    t,
                    if domain == 2 {
                        $float::MIN_POSITIVE
                    } else {
                        0.0
                    },
                );
                let inf = V::splat_t(t, $float::INFINITY);
                let aa = a.abs();
                let bb = b.abs();
                let invalid =
                    ((aa.simd_ge(lo) & aa.simd_lt(inf)) & (bb.simd_ge(lo) & bb.simd_lt(inf))).not();
                let lanes = invalid.to_array();
                if lanes.iter().any(|x| x.to_bits() != 0) {
                    let lane_mask = lanes
                        .iter()
                        .enumerate()
                        .fold(0, |m, (i, x)| m | (u16::from(x.to_bits() != 0) << i));
                    return Err(PrimitiveError {
                        kind: PrimitiveErrorKind::InvalidInput,
                        lane_mask,
                    });
                }
            }
            let nan = (O >> 2) & 3;
            let backend_zero = O & 16 != 0;
            // A normal-input proof removes all exceptional cases, so relaxed
            // WASM min/max is exact on that restricted domain too.
            #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
            if domain == 2 || (backend_zero && (nan == 2 || domain == 1)) {
                if let Some(r) =
                    crate::primitive_wasm::$float::<T, MAX, $n>(t, a.to_array(), b.to_array())
                {
                    return Ok(V::from_array_t(t, r));
                }
            }
            // Under Normal, ties have identical encodings; native satisfies
            // either NaN/zero policy with no repairs.
            let scalar =
                core::any::TypeId::of::<T>() == core::any::TypeId::of::<archmage::ScalarToken>();
            if domain == 2 {
                return Ok(if scalar || cfg!(target_arch = "wasm32") {
                    V::blend(if MAX { a.simd_gt(b) } else { a.simd_lt(b) }, a, b)
                } else if MAX {
                    a.max(b)
                } else {
                    a.min(b)
                });
            }
            #[cfg(target_arch = "wasm32")]
            if !scalar && !backend_zero && (nan == 1 || domain == 1) {
                return Ok(if MAX { a.max(b) } else { a.min(b) });
            }
            #[cfg(target_arch = "aarch64")]
            if !backend_zero && (nan == 1 || domain == 1) {
                if let Some(r) =
                    crate::primitive_neon::$float::<T, MAX, $n>(t, a.to_array(), b.to_array())
                {
                    return Ok(V::from_array_t(t, r));
                }
            }
            let mut r = if nan == 2 || domain == 1 {
                if scalar {
                    V::blend(if MAX { a.simd_gt(b) } else { a.simd_lt(b) }, a, b)
                } else if MAX {
                    a.max(b)
                } else {
                    a.min(b)
                }
            } else {
                let ordered = if MAX { a.simd_gt(b) } else { a.simd_lt(b) };
                let prefer_a = ordered | if nan == 0 { b.simd_ne(b) } else { a.simd_ne(a) };
                V::blend(prefer_a, a, b)
            };
            if !backend_zero {
                // OR and AND agree with normal equal values and order zero signs.
                let tie = if MAX { a & b } else { a | b };
                r = V::blend(a.simd_eq(b), tie, r);
            }
            Ok(r)
        }
    };
}
graph!(graph4, f32x4, F32x4Backend, f32, 4);
graph!(graph8, f32x8, F32x8Backend, f32, 8);
#[cfg(feature = "w512")]
graph!(graph16, f32x16, F32x16Backend, f32, 16);
graph!(graph64x2, f64x2, F64x2Backend, f64, 2);
graph!(graph64x4, f64x4, F64x4Backend, f64, 4);
#[cfg(feature = "w512")]
graph!(graph64x8, f64x8, F64x8Backend, f64, 8);

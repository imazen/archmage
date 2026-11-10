//! Const-configured f32/f64 min/max, with independent NaN, zero and input policies.
//!
//! Number prefers a non-NaN over either quiet or signaling NaN; Propagate returns
//! NaN if either operand is NaN. Ordered zeros choose -0 for min and +0 for max.
//! NaN payload/sign/signaling encoding and exception flags are unspecified.
//! Normal-domain graphs need no special-value repairs. AssumeValid removes the
//! input checks too; violations never cause memory unsoundness.
//!
//! Typed vector and width-specific lane entries retain their caller's token.
//! Scalar/array entries select once unless configured for Caller execution.
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

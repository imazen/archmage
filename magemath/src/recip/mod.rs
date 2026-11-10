//! Configurable f32 reciprocal and reciprocal square root.
//!
//! HIGHP reciprocal is rounded division; HIGHP rsqrt is `1 / sqrt(x)` with
//! rounding after each operation, not a correctly rounded mathematical rsqrt.
//! Non-NaN HIGHP bits agree across ISAs under round-to-nearest and gradual underflow.
//! NaN payload/sign/signaling encoding and floating exceptions are unspecified.
//!
//! LOWP/MIDP use an explicit normal operating range: 2^-125 through 2^125 in
//! magnitude, positive for rsqrt. Portable profiles share fixed integer seeds
//! and one/two non-fused Newton steps. Backend profiles may use native estimates.
//! Native estimate bits can differ across widths even with the same token;
//! use the matching width-specific lane entry for scalar tails.
mod api;
mod options;
use crate::{PrimitiveError, PrimitiveErrorKind};
pub use api::*;
pub use options::*;

macro_rules! graph {
    ($name:ident, $v:ident, $bound:ident,$n:literal) => {
        #[inline(always)]
        pub(super) fn $name<
            const O: u32,
            const RSQRT: bool,
            T: magetypes::simd::backends::$bound,
        >(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> Result<magetypes::simd::generic::$v<T>, PrimitiveError> {
            const {
                assert!(RecipOptions::valid(O), "invalid reciprocal options");
            }
            use magetypes::simd::generic::$v as V;
            let sp = |v| V::splat_t(t, v);
            if O & 1 != 0 && O & RecipOptions::ASSUME == 0 {
                let a = if RSQRT { x } else { x.abs() };
                let valid = a.simd_ge(sp(f32::from_bits(2 << 23)))
                    & a.simd_le(sp(f32::from_bits(252 << 23)));
                let bad = valid.not().bitcast_to_i32();
                if bad.any_true() {
                    let lane_mask = bad
                        .to_array()
                        .iter()
                        .enumerate()
                        .fold(0, |m, (i, &v)| m | (u16::from(v != 0) << i));
                    return Err(PrimitiveError {
                        kind: PrimitiveErrorKind::InvalidInput,
                        lane_mask,
                    });
                }
            }
            let precision = (O >> 1) & 3;
            if precision == 0 {
                return Ok(sp(1.0) / if RSQRT { x.sqrt() } else { x });
            }
            if O & RecipOptions::BACKEND != 0 {
                if precision == 2 {
                    #[cfg(target_arch = "aarch64")]
                    if let Some(r) =
                        crate::primitive_neon::estimate::<T, RSQRT, $n>(t, x.to_array())
                    {
                        return Ok(V::from_array_t(t, r));
                    }
                    return Ok(if RSQRT {
                        x.rsqrt_approx()
                    } else {
                        x.rcp_approx()
                    });
                }
                // No useful reciprocal-estimate instruction on scalar/WASM.
                // Their existing rsqrt estimate already has two fixed steps.
                #[cfg(target_arch = "aarch64")]
                if core::any::TypeId::of::<T>() == core::any::TypeId::of::<archmage::NeonToken>() {
                    return Ok(if RSQRT {
                        x.rsqrt_approx()
                    } else {
                        x.rcp_approx()
                    });
                }
                if cfg!(target_arch = "wasm32")
                    || core::any::TypeId::of::<T>()
                        == core::any::TypeId::of::<archmage::ScalarToken>()
                {
                    return Ok(if RSQRT { x.rsqrt_approx() } else { sp(1.0) / x });
                }
                return Ok(if RSQRT {
                    let y = x.rsqrt_approx();
                    let product = x * (y * y);
                    y * (-sp(0.5)).mul_add_relaxed(product, sp(1.5))
                } else {
                    let r = x.rcp_approx();
                    let e = (-x).mul_add_relaxed(r, sp(1.0));
                    r.mul_add_relaxed(e, r)
                });
            }
            let y = if RSQRT {
                x.rsqrt_approx_portable()
            } else {
                x.rcp_approx_portable()
            };
            Ok(if precision == 2 {
                y
            } else if RSQRT {
                y.rsqrt_newton_portable(x)
            } else {
                y.recip_newton_portable(x)
            })
        }
    };
}
graph!(graph4, f32x4, F32x4Convert, 4);
graph!(graph8, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
graph!(graph16, f32x16, F32x16Convert, 16);

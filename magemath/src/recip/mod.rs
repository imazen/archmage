//! Configurable f32 reciprocal and reciprocal square root.
//!
//! HIGHP reciprocal is rounded division; HIGHP rsqrt is `1 / sqrt(x)` with
//! rounding after each operation, not a correctly rounded mathematical rsqrt.
//! Non-NaN HIGHP bits agree across ISAs under round-to-nearest and gradual underflow.
//! NaN payload/sign/signaling encoding and floating exceptions are unspecified.
//!
//! LOWP/MIDP use an explicit normal operating range: 2^-125 through 2^125 in
//! magnitude, positive for rsqrt. Portable profiles share fixed integer seeds
//! and one/two canonically fused Newton steps. Backend profiles may use native estimates.
//! Native estimate bits can differ across widths even with the same token;
//! use the matching width-specific lane entry for scalar tails.
//!
//! # Presets and supported combinations
//!
//! Every preset defaults to [`crate::recip::RecipValidation::Checked`] and
//! [`crate::recip::RecipExecution::Automatic`]. Use `.with_validation(...)` or
//! `.with_execution(...)` to change those choices before `.bits()`.
//!
//! | [`crate::recip::RecipOptions`] preset | Domain | Result agreement | Accuracy / graph |
//! |---|---|---|---|
//! | `HIGHP` (also `new()`) | Full | BitExact, excluding NaN encoding | Rounded division; rsqrt is rounded sqrt then division |
//! | `PORTABLE_MIDP` | Normal | Same non-NaN bits across ISAs and widths | Fixed seed, two fused Newton steps; relative error ≤1e-4 |
//! | `PORTABLE_LOWP` | Normal | Same non-NaN bits across ISAs and widths | Fixed seed, one fused Newton step; relative error ≤6e-3 |
//! | `BACKEND_MIDP` | Normal | ISA/width-dependent bits; matching-width tails agree | Backend estimate/refinement; relative error ≤1e-4 |
//! | `BACKEND_LOWP` | Normal | ISA/width-dependent bits; matching-width tails agree | Backend approximation; relative error ≤6e-3 |
//!
//! Normal means **2^-125 ≤ abs(x) ≤ 2^125**, with **x > 0** for rsqrt.
//! Zeros, subnormals, infinities and NaNs are outside that domain. Checked
//! rejects the whole operation and reports every invalid lane. AssumeValid
//! removes those numerical checks; violations give unspecified output, never UB.
//!
//! | Domain | HIGHP | MIDP / LOWP |
//! |---|---|---|
//! | Full | Supported | Unsupported: const API compilation fails |
//! | Normal | Supported, with the same restricted-domain validation | Supported |
//!
//! BitExact / BackendDependent can be selected for either approximation grade.
//! HIGHP has one graph, so both portability choices encode identically.
//! Full-domain Checked / AssumeValid also encode identically: every input is
//! supported. Normal-domain validation choices remain distinct. Both execution
//! choices are supported; explicit vector and lane entries retain their supplied
//! token, while scalar/array Caller entries stay scalar. Without `std`, optional
//! x86 features need a caller proof or compile-time target support.
//!
//! BitExact uses hardware FMA when available; it does not force software math.
//! Its existing fallback is exact software FMA. Bounded-f64 evaluation and
//! canonical RN14/RN8 are research implementations, not selectable API presets.
//! Lower precision and omitted checks do not guarantee a speedup on every ISA.
//!
//! # Const configuration and matching tails
//!
//! ```
//! use archmage::ScalarToken;
//! use magetypes::simd::generic::f32x4;
//! use magemath::recip::{
//!     RecipExecution, RecipOptions, RecipValidation,
//!     try_recip_f32x4_with, try_recip_f32x4_lane_with,
//! };
//! const FLAGS: u32 = RecipOptions::PORTABLE_MIDP
//!     .with_validation(RecipValidation::AssumeValid)
//!     .with_execution(RecipExecution::Caller)
//!     .bits();
//! // All lanes meet the selected Normal-domain precondition.
//! let t = ScalarToken;
//! let x = f32x4::from_array_t(t, [1.0, 2.0, 3.0, 4.0]);
//! let y = try_recip_f32x4_with::<FLAGS, _>(t, x).unwrap();
//! let tail = try_recip_f32x4_lane_with::<FLAGS, _>(t, 3.0).unwrap();
//! assert_eq!(y.to_array()[2].to_bits(), tail.to_bits());
//! ```
#![doc = include_str!("permutations.md")]
// BEGIN GENERATED TIMING DOCUMENTS
#![doc = include_str!("timings/mac-neon-recip.md")]
#![doc = include_str!("timings/mac-neon-rsqrt.md")]
#![doc = include_str!("timings/i265-v3-recip.md")]
#![doc = include_str!("timings/i265-v3-rsqrt.md")]
#![doc = include_str!("timings/dev-scalar-recip.md")]
#![doc = include_str!("timings/dev-scalar-rsqrt.md")]
#![doc = include_str!("timings/dev-v3-recip.md")]
#![doc = include_str!("timings/dev-v3-rsqrt.md")]
#![doc = include_str!("timings/dev-v4-recip.md")]
#![doc = include_str!("timings/dev-v4-rsqrt.md")]
#![doc = include_str!("timings/wasm-strict-recip.md")]
#![doc = include_str!("timings/wasm-strict-rsqrt.md")]
// END GENERATED TIMING DOCUMENTS
mod api;
mod options;
use crate::{PrimitiveError, PrimitiveErrorKind};
pub use api::*;
pub use options::*;

// Arithmetic is selected once before entering either Newton graph.
const NATIVE: u8 = 0;
const SOFTWARE: u8 = 1;
#[cfg(all(feature = "std", target_arch = "x86_64", not(miri)))]
const STD_FMA: u8 = 3;
#[cfg(target_arch = "wasm32")]
const WASM_SOFTWARE: u8 = 2;

#[inline(always)]
fn native_fusion<T: archmage::SimdToken>(_: T) -> bool {
    let id = core::any::TypeId::of::<T>();
    #[cfg(target_arch = "x86_64")]
    {
        if id == core::any::TypeId::of::<archmage::X64V3Token>() {
            return true;
        }
        #[cfg(feature = "avx512")]
        if id == core::any::TypeId::of::<archmage::X64V4Token>()
            || id == core::any::TypeId::of::<archmage::X64V4xToken>()
        {
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if id == core::any::TypeId::of::<archmage::NeonToken>() {
        return true;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    if id == core::any::TypeId::of::<archmage::Wasm128Token>() {
        use archmage::SimdToken;
        return archmage::Wasm128RelaxedToken::summon().is_some()
            && crate::wasm_fma::powf_f32_fuses();
    }
    let _ = id;
    false
}

macro_rules! graph {
    ($name:ident,$core:ident,$v:ident,$int:ident,$bound:ident,$n:literal) => {
        #[inline(always)]
        pub(super) fn $name<
            const O: u32,
            const RSQRT: bool,
            T: magetypes::simd::backends::$bound,
        >(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> Result<magetypes::simd::generic::$v<T>, PrimitiveError> {
            // HIGHP and backend-estimate graphs do not need the portable FMA selector.
            if O & 6 == 0 || O & RecipOptions::BACKEND != 0 || native_fusion(t) {
                return $core::<O, RSQRT, NATIVE, T>(t, x);
            }
            // FMA3 can exist without all V3 prerequisites. No additional proof
            // token is fabricated: std's correctly rounded primitive owns its
            // platform dispatch, selected here once for the complete graph.
            #[cfg(all(feature = "std", target_arch = "x86_64", not(miri)))]
            if O & RecipOptions::CALLER == 0
                && core::any::TypeId::of::<T>() == core::any::TypeId::of::<archmage::ScalarToken>()
                && std::is_x86_feature_detected!("fma")
            {
                return $core::<O, RSQRT, STD_FMA, T>(t, x);
            }
            #[cfg(target_arch = "wasm32")]
            if core::any::TypeId::of::<T>() == core::any::TypeId::of::<archmage::Wasm128Token>() {
                return $core::<O, RSQRT, WASM_SOFTWARE, T>(t, x);
            }
            $core::<O, RSQRT, SOFTWARE, T>(t, x)
        }
        #[inline(always)]
        fn $core<
            const O: u32,
            const RSQRT: bool,
            const ARITHMETIC: u8,
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
            let madd = |a: V<T>, b: V<T>, c: V<T>| {
                if ARITHMETIC == NATIVE {
                    #[cfg(target_arch = "wasm32")]
                    if let Some(proof) =
                        (&t as &dyn core::any::Any).downcast_ref::<archmage::Wasm128Token>()
                    {
                        // The operation boundary has already proved fusion and
                        // observed token testing. Do not select again per FMA.
                        return V::from_array_t(
                            t,
                            crate::powf::wasm::hardware_lanes(
                                *proof,
                                a.to_array(),
                                b.to_array(),
                                c.to_array(),
                            ),
                        );
                    }
                    return a.mul_add_relaxed(b, c);
                }
                #[cfg(target_arch = "wasm32")]
                if ARITHMETIC == WASM_SOFTWARE {
                    let any = &t as &dyn core::any::Any;
                    let proof = *any
                        .downcast_ref::<archmage::Wasm128Token>()
                        .expect("private WASM FMA mode requires its proof");
                    return V::from_array_t(
                        t,
                        crate::powf::wasm::software_lanes(
                            proof,
                            a.to_array(),
                            b.to_array(),
                            c.to_array(),
                        ),
                    );
                }
                #[cfg(all(feature = "std", target_arch = "x86_64", not(miri)))]
                if ARITHMETIC == STD_FMA {
                    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
                    return V::from_array_t(t, core::array::from_fn(|i| a[i].mul_add(b[i], c[i])));
                }
                // Scalar f32 fused arithmetic has no runtime feature probe.
                // Compile-proven hardware is still allowed by this primitive.
                a.mul_add_fused(b, c)
            };
            use magetypes::simd::generic::$int as I;
            let bits = x.bitcast_to_i32();
            let seed = if RSQRT {
                I::splat_t(t, 0x5f3759df) - bits.shr_logical_const::<1>()
            } else {
                I::splat_t(t, 0x7ef127ea) - bits
            };
            let mut y = seed.bitcast_to_f32();
            for _ in 0..if precision == 2 { 1 } else { 2 } {
                y = if RSQRT {
                    // Explicit graph: three multiplications and one fused add.
                    // Retain left association in the product to avoid y*y
                    // overflowing at the low end of the accepted range.
                    let product = (sp(0.5) * x) * y;
                    y * madd(-product, y, sp(1.5))
                } else {
                    y * madd(-x, y, sp(2.0))
                };
            }
            Ok(y)
        }
    };
}
graph!(graph4, core4, f32x4, i32x4, F32x4Convert, 4);
graph!(graph8, core8, f32x8, i32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
graph!(graph16, core16, f32x16, i32x16, F32x16Convert, 16);

#[cfg(all(
    test,
    feature = "std",
    target_arch = "wasm32",
    target_feature = "relaxed-simd"
))]
mod tests {
    use super::*;
    use archmage::{SimdToken, Wasm128RelaxedToken, Wasm128Token};
    #[test]
    fn fused_provider_respects_probe_and_token_suppression() {
        let _guard = archmage::testing::lock_token_testing();
        let t = Wasm128Token::summon().expect("SIMD module");
        let fuses = crate::wasm_fma::powf_f32_fuses();
        match std::env::var("ARCHMAGE_EXPECT_RELAXED_FUSION").as_deref() {
            Ok("yes") => assert!(fuses),
            Ok("no") => assert!(!fuses),
            _ => {}
        }
        assert_eq!(native_fusion(t), fuses);
        use magetypes::simd::generic::f32x8 as V;
        const O: u32 = RecipOptions::PORTABLE_MIDP.bits();
        let input = V::from_array_t(
            t,
            [
                1.1,
                3.0,
                f32::from_bits(0x3ff12999),
                0.125,
                2.25,
                7.0,
                16.0,
                1.0,
            ],
        );
        let rc = try_recip_f32x8_with::<O, _>(t, input)
            .unwrap()
            .to_array()
            .map(f32::to_bits);
        let rs = try_rsqrt_f32x8_with::<O, _>(t, input)
            .unwrap()
            .to_array()
            .map(f32::to_bits);
        if std::env::var("ARCHMAGE_EXPECT_TESTABLE_DISPATCH").as_deref() == Ok("yes") {
            assert!(Wasm128RelaxedToken::compiled_with().is_none());
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(!native_fusion(t));
            assert_eq!(
                rc,
                try_recip_f32x8_with::<O, _>(t, input)
                    .unwrap()
                    .to_array()
                    .map(f32::to_bits)
            );
            assert_eq!(
                rs,
                try_rsqrt_f32x8_with::<O, _>(t, input)
                    .unwrap()
                    .to_array()
                    .map(f32::to_bits)
            );
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(false).unwrap();
            assert_eq!(native_fusion(t), fuses);
        }
    }
}

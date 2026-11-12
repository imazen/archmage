//! Register-direct WASM min/max kernels:
//! - With `relaxed-simd` and `Wasm128RelaxedToken`: `f32x4_relaxed_min`/`max`, `f64x2_relaxed_min`/`max`.
//! - Baseline `simd128` (`Wasm128Token`): `f32x4_pmin`/`pmax`, `f64x2_pmin`/`pmax` (`b < a ? b : a` / `a < b ? b : a`),
//!   which lower to a single native instruction (`vminps`/`vmaxps`) on x86-64 engines without `relaxed-simd`
//!   and satisfy Normal, Finite+BackendZero, and Relaxed policies identically.
use archmage::Wasm128Token;
#[cfg(target_feature = "relaxed-simd")]
use archmage::{SimdToken, Wasm128RelaxedToken};
use core::any::Any;
use core::arch::wasm32::{self as w, v128};
use magetypes::simd::backends::{F32x4Backend, F32x8Backend, F64x2Backend, F64x4Backend};
#[cfg(feature = "w512")]
use magetypes::simd::backends::{F32x16Backend, F64x8Backend};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{f32x16, f64x8};

#[inline(always)]
fn cast_same<A: 'static + Copy, B: 'static + Copy>(val: A) -> Option<B> {
    (&val as &dyn Any).downcast_ref::<B>().copied()
}

#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn relaxed_f32_op<const MAX: bool>(a: v128, b: v128) -> v128 {
    if MAX {
        w::f32x4_relaxed_max(a, b)
    } else {
        w::f32x4_relaxed_min(a, b)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn relaxed_f64_op<const MAX: bool>(a: v128, b: v128) -> v128 {
    if MAX {
        w::f64x2_relaxed_max(a, b)
    } else {
        w::f64x2_relaxed_min(a, b)
    }
}

#[cfg_attr(target_feature = "simd128", inline(always))]
#[cfg_attr(not(target_feature = "simd128"), archmage::arcane(suppress_const_test))]
fn pminmax_f32_op<const MAX: bool>(_t: Wasm128Token, a: v128, b: v128) -> v128 {
    if MAX {
        w::f32x4_pmax(a, b)
    } else {
        w::f32x4_pmin(a, b)
    }
}

#[cfg_attr(target_feature = "simd128", inline(always))]
#[cfg_attr(not(target_feature = "simd128"), archmage::arcane(suppress_const_test))]
fn pminmax_f64_op<const MAX: bool>(_t: Wasm128Token, a: v128, b: v128) -> v128 {
    if MAX {
        w::f64x2_pmax(a, b)
    } else {
        w::f64x2_pmin(a, b)
    }
}

#[inline(always)]
fn fast_f32_v128<const MAX: bool>(t: Wasm128Token, a: v128, b: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    if Wasm128RelaxedToken::summon().is_some() {
        return relaxed_f32_op::<MAX>(a, b);
    }
    pminmax_f32_op::<MAX>(t, a, b)
}

#[inline(always)]
fn fast_f64_v128<const MAX: bool>(t: Wasm128Token, a: v128, b: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    if Wasm128RelaxedToken::summon().is_some() {
        return relaxed_f64_op::<MAX>(a, b);
    }
    pminmax_f64_op::<MAX>(t, a, b)
}

#[inline(always)]
pub(crate) fn minmax_f32x4<const MAX: bool, T: F32x4Backend>(
    t: T,
    a: f32x4<T>,
    b: f32x4<T>,
) -> Option<f32x4<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let va = cast_same::<_, f32x4<Wasm128Token>>(a)?.into_repr();
    let vb = cast_same::<_, f32x4<Wasm128Token>>(b)?.into_repr();
    cast_same(f32x4::from_repr_t(*wt, fast_f32_v128::<MAX>(*wt, va, vb)))
}

#[inline(always)]
pub(crate) fn minmax_f32x8<const MAX: bool, T: F32x8Backend>(
    t: T,
    a: f32x8<T>,
    b: f32x8<T>,
) -> Option<f32x8<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1] = cast_same::<_, f32x8<Wasm128Token>>(a)?.into_repr();
    let [b0, b1] = cast_same::<_, f32x8<Wasm128Token>>(b)?.into_repr();
    cast_same(f32x8::from_repr_t(
        *wt,
        [
            fast_f32_v128::<MAX>(*wt, a0, b0),
            fast_f32_v128::<MAX>(*wt, a1, b1),
        ],
    ))
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn minmax_f32x16<const MAX: bool, T: F32x16Backend>(
    t: T,
    a: f32x16<T>,
    b: f32x16<T>,
) -> Option<f32x16<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f32x16<Wasm128Token>>(a)?.into_repr();
    let [b0, b1, b2, b3] = cast_same::<_, f32x16<Wasm128Token>>(b)?.into_repr();
    cast_same(f32x16::from_repr_t(
        *wt,
        [
            fast_f32_v128::<MAX>(*wt, a0, b0),
            fast_f32_v128::<MAX>(*wt, a1, b1),
            fast_f32_v128::<MAX>(*wt, a2, b2),
            fast_f32_v128::<MAX>(*wt, a3, b3),
        ],
    ))
}

#[inline(always)]
pub(crate) fn minmax_f64x2<const MAX: bool, T: F64x2Backend>(
    t: T,
    a: f64x2<T>,
    b: f64x2<T>,
) -> Option<f64x2<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let va = cast_same::<_, f64x2<Wasm128Token>>(a)?.into_repr();
    let vb = cast_same::<_, f64x2<Wasm128Token>>(b)?.into_repr();
    cast_same(f64x2::from_repr_t(*wt, fast_f64_v128::<MAX>(*wt, va, vb)))
}

#[inline(always)]
pub(crate) fn minmax_f64x4<const MAX: bool, T: F64x4Backend>(
    t: T,
    a: f64x4<T>,
    b: f64x4<T>,
) -> Option<f64x4<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1] = cast_same::<_, f64x4<Wasm128Token>>(a)?.into_repr();
    let [b0, b1] = cast_same::<_, f64x4<Wasm128Token>>(b)?.into_repr();
    cast_same(f64x4::from_repr_t(
        *wt,
        [
            fast_f64_v128::<MAX>(*wt, a0, b0),
            fast_f64_v128::<MAX>(*wt, a1, b1),
        ],
    ))
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn minmax_f64x8<const MAX: bool, T: F64x8Backend>(
    t: T,
    a: f64x8<T>,
    b: f64x8<T>,
) -> Option<f64x8<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f64x8<Wasm128Token>>(a)?.into_repr();
    let [b0, b1, b2, b3] = cast_same::<_, f64x8<Wasm128Token>>(b)?.into_repr();
    cast_same(f64x8::from_repr_t(
        *wt,
        [
            fast_f64_v128::<MAX>(*wt, a0, b0),
            fast_f64_v128::<MAX>(*wt, a1, b1),
            fast_f64_v128::<MAX>(*wt, a2, b2),
            fast_f64_v128::<MAX>(*wt, a3, b3),
        ],
    ))
}

/// Exposed for token-suppression unit tests under `relaxed-simd`.
#[cfg(all(test, feature = "std", target_feature = "relaxed-simd"))]
#[inline(always)]
fn relaxed_only_f32x4<const MAX: bool>(
    t: Wasm128Token,
    a: f32x4<Wasm128Token>,
    b: f32x4<Wasm128Token>,
) -> Option<f32x4<Wasm128Token>> {
    Wasm128RelaxedToken::summon()?;
    Some(f32x4::from_repr_t(
        t,
        relaxed_f32_op::<MAX>(a.into_repr(), b.into_repr()),
    ))
}

#[cfg(all(test, feature = "std", target_feature = "relaxed-simd"))]
mod tests {
    use super::*;
    #[test]
    fn testable_relaxed_selection() {
        let _guard = archmage::testing::lock_token_testing();
        let t = Wasm128Token::summon().expect("SIMD module");
        let a = f32x4::from_array_t(t, [1.; 4]);
        let b = f32x4::from_array_t(t, [2.; 4]);
        assert!(relaxed_only_f32x4::<false>(t, a, b).is_some());
        assert_eq!(
            minmax_f32x4::<false, _>(t, a, b).unwrap().to_array(),
            [1.; 4]
        );
        if std::env::var("ARCHMAGE_EXPECT_TESTABLE_DISPATCH").as_deref() == Ok("yes") {
            assert!(
                Wasm128RelaxedToken::compiled_with().is_none(),
                "testable dispatch must expose suppression"
            );
        }
        if Wasm128RelaxedToken::compiled_with().is_none() {
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(relaxed_only_f32x4::<false>(t, a, b).is_none());
            // Baseline pmin fallback remains active and exact.
            assert_eq!(
                minmax_f32x4::<false, _>(t, a, b).unwrap().to_array(),
                [1.; 4]
            );
            Wasm128RelaxedToken::dangerously_disable_token_process_wide(false).unwrap();
            assert!(relaxed_only_f32x4::<false>(t, a, b).is_some());
        }
    }
}

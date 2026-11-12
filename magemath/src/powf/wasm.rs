//! Packed software and hardware FMA in WASM registers, entered after whole-operation selection.
use archmage::Wasm128Token;
use core::any::Any;
use core::arch::wasm32::{self as w, v128};
#[cfg(feature = "w512")]
use magetypes::simd::backends::F32x16Backend;
use magetypes::simd::backends::{F32x4Backend, F32x8Backend};
#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;
use magetypes::simd::generic::{f32x4, f32x8};

#[inline(always)]
fn cast_same<A: 'static + Copy, B: 'static + Copy>(val: A) -> Option<B> {
    (&val as &dyn Any).downcast_ref::<B>().copied()
}

#[cfg_attr(target_feature = "simd128", inline(always))]
#[cfg_attr(not(target_feature = "simd128"), archmage::arcane(suppress_const_test))]
fn step_v128<const SOFTWARE: bool>(token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if SOFTWARE {
        crate::wasm_fma::powf_soft_f32x4(token, a, b, c)
    } else {
        crate::wasm_fma::madd_f32x4(token, a, b, c)
    }
}

#[inline(always)]
pub(crate) fn fma_f32x4<const SOFTWARE: bool, T: F32x4Backend>(
    t: T,
    a: f32x4<T>,
    b: f32x4<T>,
    c: f32x4<T>,
) -> Option<f32x4<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let va = cast_same::<_, f32x4<Wasm128Token>>(a)?.into_repr();
    let vb = cast_same::<_, f32x4<Wasm128Token>>(b)?.into_repr();
    let vc = cast_same::<_, f32x4<Wasm128Token>>(c)?.into_repr();
    cast_same(f32x4::from_repr_t(
        *wt,
        step_v128::<SOFTWARE>(*wt, va, vb, vc),
    ))
}

#[inline(always)]
pub(crate) fn fma_f32x8<const SOFTWARE: bool, T: F32x8Backend>(
    t: T,
    a: f32x8<T>,
    b: f32x8<T>,
    c: f32x8<T>,
) -> Option<f32x8<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1] = cast_same::<_, f32x8<Wasm128Token>>(a)?.into_repr();
    let [b0, b1] = cast_same::<_, f32x8<Wasm128Token>>(b)?.into_repr();
    let [c0, c1] = cast_same::<_, f32x8<Wasm128Token>>(c)?.into_repr();
    cast_same(f32x8::from_repr_t(
        *wt,
        [
            step_v128::<SOFTWARE>(*wt, a0, b0, c0),
            step_v128::<SOFTWARE>(*wt, a1, b1, c1),
        ],
    ))
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn fma_f32x16<const SOFTWARE: bool, T: F32x16Backend>(
    t: T,
    a: f32x16<T>,
    b: f32x16<T>,
    c: f32x16<T>,
) -> Option<f32x16<T>> {
    let wt = (&t as &dyn Any).downcast_ref::<Wasm128Token>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f32x16<Wasm128Token>>(a)?.into_repr();
    let [b0, b1, b2, b3] = cast_same::<_, f32x16<Wasm128Token>>(b)?.into_repr();
    let [c0, c1, c2, c3] = cast_same::<_, f32x16<Wasm128Token>>(c)?.into_repr();
    cast_same(f32x16::from_repr_t(
        *wt,
        [
            step_v128::<SOFTWARE>(*wt, a0, b0, c0),
            step_v128::<SOFTWARE>(*wt, a1, b1, c1),
            step_v128::<SOFTWARE>(*wt, a2, b2, c2),
            step_v128::<SOFTWARE>(*wt, a3, b3, c3),
        ],
    ))
}

#[cfg_attr(target_feature = "simd128", inline(always))]
#[cfg_attr(not(target_feature = "simd128"), archmage::arcane(suppress_const_test))]
fn lanes<const SOFTWARE: bool, const N: usize>(
    token: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    assert_eq!(
        N % 4,
        0,
        "private packed software adapter requires whole vectors"
    );
    let mut output = [0.0; N];
    for (((a, b), c), out) in a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .zip(c.as_chunks::<4>().0)
        .zip(output.as_chunks_mut::<4>().0)
    {
        let va = f32x4::from_array_t(token, *a).into_repr();
        let vb = f32x4::from_array_t(token, *b).into_repr();
        let vc = f32x4::from_array_t(token, *c).into_repr();
        let r = step_v128::<SOFTWARE>(token, va, vb, vc);
        *out = f32x4::from_repr_t(token, r).to_array();
    }
    let _ = w::f32x4_extract_lane::<0>;
    output
}

/// Entered only after a whole-operation decision; neither form probes per FMA.
#[inline(always)]
pub(crate) fn software_lanes<const N: usize>(
    t: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    lanes::<true, N>(t, a, b, c)
}
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn hardware_lanes<const N: usize>(
    t: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    lanes::<false, N>(t, a, b, c)
}

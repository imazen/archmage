//! Packed policies share scalar preparation, fallback and output semantics.
use magetypes::simd::{
    backends::{F32x8Convert, F64x4Backend},
    generic::f32x8,
};
use zenmath::__pow_policy as shared;
#[inline(always)]
fn core<T: F64x4Backend>(t: T, x: [f32; 8], y: [f32; 8]) -> [f32; 8] {
    let lo = crate::correct::pow(
        t,
        core::array::from_fn(|i| x[i]),
        core::array::from_fn(|i| y[i]),
    );
    let hi = crate::correct::pow(
        t,
        core::array::from_fn(|i| x[i + 4]),
        core::array::from_fn(|i| y[i + 4]),
    );
    core::array::from_fn(|i| if i < 4 { lo[i] } else { hi[i - 4] })
}
#[inline(always)]
pub(crate) fn array<T: F32x8Convert + F64x4Backend, const P: u64>(
    t: T,
    x: [f32; 8],
    y: [f32; 8],
) -> [f32; 8] {
    const {
        shared::validate(P);
    }
    if P == 0 {
        return core(t, x, y);
    }
    let prepared = core::array::from_fn::<_, 8, _>(|i| shared::prepare::<P>(x[i], y[i]));
    let xx = prepared.map(|p| p.x);
    let yy = prepared.map(|p| p.y);
    let out = if P & 3 == 0 {
        core(t, xx, yy)
    } else if P & (1 << 2) == 0 {
        approximate::<crate::bodies::Fused, _, P>(t, prepared)
    } else {
        approximate::<crate::bodies::Unfused, _, P>(t, prepared)
    };
    core::array::from_fn(|i| shared::finish::<P>(prepared[i], out[i]))
}
#[inline(always)]
fn approximate<S: crate::bodies::Step, T: F32x8Convert, const P: u64>(
    t: T,
    p: [shared::Prepared; 8],
) -> [f32; 8] {
    let precondition = (P >> 9) & 3 == 2;
    let x = f32x8::from_array_t(t, p.map(|v| if precondition { v.x } else { v.x.abs() }));
    let y = f32x8::from_array_t(t, p.map(|v| v.y));
    let log = if P & 3 == 2 {
        crate::bodies::log2_lowp::<S, _>(t, x)
    } else {
        crate::bodies::log2_midp::<S, _>(t, x)
    };
    let z = log * y;
    let out = if P & 3 == 2 {
        crate::bodies::exp2_lowp::<S, _>(t, z)
    } else {
        crate::bodies::exp2_midp::<S, _>(t, z)
    };
    let z = z.to_array();
    let out = out.to_array();
    core::array::from_fn(|i| {
        if precondition {
            // The existing lowp formula explicitly returns +0 for either zero
            // base, including n==0. Preserve this defined release behavior.
            if P & 3 == 2 && p[i].x == 0.0 {
                0.0
            } else {
                out[i]
            }
        } else if shared::working_input(p[i]) && (-126.0..=127.5).contains(&z[i]) {
            shared::mathematical_sign(p[i], out[i])
        } else {
            zenmath::scalar::powf(p[i].x, p[i].y)
        }
    })
}

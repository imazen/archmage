//! Propagating IEEE min/max and reciprocal/rsqrt estimates in native NEON registers.
//! Uses `into_repr` / `from_repr_t` so single-instruction kernels never bounce
//! through `[f32; N]` stack arrays or scalar lane insert/extract sequences.
use archmage::{NeonToken, SimdToken};
use core::any::Any;
use core::arch::aarch64::{self as n, float32x4_t, float64x2_t};
use magetypes::simd::backends::{F32x4Backend, F32x8Backend, F64x2Backend, F64x4Backend};
#[cfg(feature = "w512")]
use magetypes::simd::backends::{F32x16Backend, F64x8Backend};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{f32x16, f64x8};

mod kernel {
    use super::*;

    #[archmage::arcane(suppress_const_test)]
    pub(super) fn minmax_f32<const MAX: bool>(
        _t: NeonToken,
        a: float32x4_t,
        b: float32x4_t,
    ) -> float32x4_t {
        if MAX {
            n::vmaxq_f32(a, b)
        } else {
            n::vminq_f32(a, b)
        }
    }

    #[archmage::arcane(suppress_const_test)]
    pub(super) fn minmax_f64<const MAX: bool>(
        _t: NeonToken,
        a: float64x2_t,
        b: float64x2_t,
    ) -> float64x2_t {
        if MAX {
            n::vmaxq_f64(a, b)
        } else {
            n::vminq_f64(a, b)
        }
    }

    #[archmage::arcane(suppress_const_test)]
    pub(super) fn est_f32<const RSQRT: bool>(_t: NeonToken, a: float32x4_t) -> float32x4_t {
        if RSQRT {
            n::vrsqrteq_f32(a)
        } else {
            n::vrecpeq_f32(a)
        }
    }
}

#[inline(always)]
fn cast_same<A: 'static + Copy, B: 'static + Copy>(val: A) -> Option<B> {
    (&val as &dyn Any).downcast_ref::<B>().copied()
}

#[inline(always)]
pub(crate) fn minmax_f32x4<const MAX: bool, T: F32x4Backend>(
    t: T,
    a: f32x4<T>,
    b: f32x4<T>,
) -> Option<f32x4<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let va: f32x4<NeonToken> = cast_same(a)?;
    let vb: f32x4<NeonToken> = cast_same(b)?;
    let out = f32x4::from_repr_t(
        *nt,
        kernel::minmax_f32::<MAX>(*nt, va.into_repr(), vb.into_repr()),
    );
    cast_same(out)
}

#[inline(always)]
pub(crate) fn minmax_f32x8<const MAX: bool, T: F32x8Backend>(
    t: T,
    a: f32x8<T>,
    b: f32x8<T>,
) -> Option<f32x8<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1] = cast_same::<_, f32x8<NeonToken>>(a)?.into_repr();
    let [b0, b1] = cast_same::<_, f32x8<NeonToken>>(b)?.into_repr();
    let out = f32x8::from_repr_t(
        *nt,
        [
            kernel::minmax_f32::<MAX>(*nt, a0, b0),
            kernel::minmax_f32::<MAX>(*nt, a1, b1),
        ],
    );
    cast_same(out)
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn minmax_f32x16<const MAX: bool, T: F32x16Backend>(
    t: T,
    a: f32x16<T>,
    b: f32x16<T>,
) -> Option<f32x16<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f32x16<NeonToken>>(a)?.into_repr();
    let [b0, b1, b2, b3] = cast_same::<_, f32x16<NeonToken>>(b)?.into_repr();
    let out = f32x16::from_repr_t(
        *nt,
        [
            kernel::minmax_f32::<MAX>(*nt, a0, b0),
            kernel::minmax_f32::<MAX>(*nt, a1, b1),
            kernel::minmax_f32::<MAX>(*nt, a2, b2),
            kernel::minmax_f32::<MAX>(*nt, a3, b3),
        ],
    );
    cast_same(out)
}

#[inline(always)]
pub(crate) fn minmax_f64x2<const MAX: bool, T: F64x2Backend>(
    t: T,
    a: f64x2<T>,
    b: f64x2<T>,
) -> Option<f64x2<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let va: f64x2<NeonToken> = cast_same(a)?;
    let vb: f64x2<NeonToken> = cast_same(b)?;
    let out = f64x2::from_repr_t(
        *nt,
        kernel::minmax_f64::<MAX>(*nt, va.into_repr(), vb.into_repr()),
    );
    cast_same(out)
}

#[inline(always)]
pub(crate) fn minmax_f64x4<const MAX: bool, T: F64x4Backend>(
    t: T,
    a: f64x4<T>,
    b: f64x4<T>,
) -> Option<f64x4<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1] = cast_same::<_, f64x4<NeonToken>>(a)?.into_repr();
    let [b0, b1] = cast_same::<_, f64x4<NeonToken>>(b)?.into_repr();
    let out = f64x4::from_repr_t(
        *nt,
        [
            kernel::minmax_f64::<MAX>(*nt, a0, b0),
            kernel::minmax_f64::<MAX>(*nt, a1, b1),
        ],
    );
    cast_same(out)
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn minmax_f64x8<const MAX: bool, T: F64x8Backend>(
    t: T,
    a: f64x8<T>,
    b: f64x8<T>,
) -> Option<f64x8<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f64x8<NeonToken>>(a)?.into_repr();
    let [b0, b1, b2, b3] = cast_same::<_, f64x8<NeonToken>>(b)?.into_repr();
    let out = f64x8::from_repr_t(
        *nt,
        [
            kernel::minmax_f64::<MAX>(*nt, a0, b0),
            kernel::minmax_f64::<MAX>(*nt, a1, b1),
            kernel::minmax_f64::<MAX>(*nt, a2, b2),
            kernel::minmax_f64::<MAX>(*nt, a3, b3),
        ],
    );
    cast_same(out)
}

#[inline(always)]
pub(crate) fn estimate_f32x4<const RSQRT: bool, T: F32x4Backend>(
    t: T,
    a: f32x4<T>,
) -> Option<f32x4<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let va: f32x4<NeonToken> = cast_same(a)?;
    let out = f32x4::from_repr_t(*nt, kernel::est_f32::<RSQRT>(*nt, va.into_repr()));
    cast_same(out)
}

#[inline(always)]
pub(crate) fn estimate_f32x8<const RSQRT: bool, T: F32x8Backend>(
    t: T,
    a: f32x8<T>,
) -> Option<f32x8<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1] = cast_same::<_, f32x8<NeonToken>>(a)?.into_repr();
    let out = f32x8::from_repr_t(
        *nt,
        [
            kernel::est_f32::<RSQRT>(*nt, a0),
            kernel::est_f32::<RSQRT>(*nt, a1),
        ],
    );
    cast_same(out)
}

#[cfg(feature = "w512")]
#[inline(always)]
pub(crate) fn estimate_f32x16<const RSQRT: bool, T: F32x16Backend>(
    t: T,
    a: f32x16<T>,
) -> Option<f32x16<T>> {
    let nt = (&t as &dyn Any).downcast_ref::<NeonToken>()?;
    let [a0, a1, a2, a3] = cast_same::<_, f32x16<NeonToken>>(a)?.into_repr();
    let out = f32x16::from_repr_t(
        *nt,
        [
            kernel::est_f32::<RSQRT>(*nt, a0),
            kernel::est_f32::<RSQRT>(*nt, a1),
            kernel::est_f32::<RSQRT>(*nt, a2),
            kernel::est_f32::<RSQRT>(*nt, a3),
        ],
    );
    cast_same(out)
}

#[allow(dead_code)]
const _: fn() = || {
    fn _keep_trait<T: SimdToken>() {}
};

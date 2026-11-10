use crate::Case;
use algebraic_magetypes_prototype::{AlgebraicOps, AlgebraicOpsWithToken, AlgebraicReduce};
use archmage::prelude::*;
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = (((((f32x8::splat_t(t, 1.0 / 6.0) * a) + f32x8::splat_t(t, 0.5)) * a)
            + f32x8::splat_t(t, 1.0))
            * a)
            + f32x8::splat_t(t, 1.0);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_index_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = f32x8::splat_t(t, 1.0 / 6.0)
            .algebraic_mul(a)
            .algebraic_add(f32x8::splat_t(t, 0.5))
            .algebraic_mul(a)
            .algebraic_add(f32x8::splat_t(t, 1.0))
            .algebraic_mul(a)
            .algebraic_add(f32x8::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_token_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = f32x8::splat_t(t, 1.0 / 6.0)
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f32x8::splat_t(t, 0.5))
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f32x8::splat_t(t, 1.0))
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f32x8::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_map_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = map_add_f32(
            t,
            map_mul_f32(
                t,
                map_add_f32(
                    t,
                    map_mul_f32(
                        t,
                        map_add_f32(
                            t,
                            map_mul_f32(t, f32x8::splat_t(t, 1.0 / 6.0), a),
                            f32x8::splat_t(t, 0.5),
                        ),
                        a,
                    ),
                    f32x8::splat_t(t, 1.0),
                ),
                a,
            ),
            f32x8::splat_t(t, 1.0),
        );
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_auto_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for (o, &a) in out.iter_mut().zip(x) {
        *o = (1.0_f32 / 6.0)
            .algebraic_mul(a)
            .algebraic_add(0.5)
            .algebraic_mul(a)
            .algebraic_add(1.0)
            .algebraic_mul(a)
            .algebraic_add(1.0);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_fused_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let _ = b;
        let r = f32x8::splat_t(t, 1.0 / 6.0)
            .mul_add_fused(a, f32x8::splat_t(t, 0.5))
            .mul_add_fused(a, f32x8::splat_t(t, 1.0))
            .mul_add_fused(a, f32x8::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum += a * b;
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_index_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum = sum.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_token_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum = sum.algebraic_add_t(t, a.algebraic_mul_t(t, b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_map_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum = map_add_f32(t, sum, map_mul_f32(t, a, b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_auto_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut r = 0.0_f32;
    for (&a, &b) in x.iter().zip(y) {
        r = r.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = r;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_fused_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum = a.mul_add_fused(b, sum);
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_index_ordinary_reduce_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let mut sum = f32x8::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0) {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        sum = sum.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn div_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = a / b;
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn div_index_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = a.algebraic_div(b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn div_token_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = a.algebraic_div_t(t, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn div_map_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = map_div_f32(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn div_auto_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        *o = a.algebraic_div(b);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn rem_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = map_rem_ordinary_f32(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn rem_index_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = a.algebraic_rem(b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn rem_token_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = a.algebraic_rem_t(t, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn rem_map_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        let a = f32x8::load_t(t, a);
        let b = f32x8::load_t(t, b);
        let r = map_rem_f32(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn rem_auto_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        *o = a.algebraic_rem(b);
    }
}
#[inline(always)]
fn map_add_f32<T: magetypes::simd::backends::F32x8Backend>(
    t: T,
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f32x8::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_add(b[i])),
    )
}
#[inline(always)]
fn map_mul_f32<T: magetypes::simd::backends::F32x8Backend>(
    t: T,
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f32x8::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
    )
}
#[inline(always)]
fn map_div_f32<T: magetypes::simd::backends::F32x8Backend>(
    t: T,
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f32x8::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
    )
}
#[inline(always)]
fn map_rem_f32<T: magetypes::simd::backends::F32x8Backend>(
    t: T,
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f32x8::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
    )
}
#[inline(always)]
fn map_rem_ordinary_f32<T: magetypes::simd::backends::F32x8Backend>(
    t: T,
    a: magetypes::simd::generic::f32x8<T>,
    b: magetypes::simd::generic::f32x8<T>,
) -> magetypes::simd::generic::f32x8<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f32x8::from_array_t(t, core::array::from_fn(|i| a[i] % b[i]))
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let _ = b;
        let r = (((((f64x4::splat_t(t, 1.0 / 6.0) * a) + f64x4::splat_t(t, 0.5)) * a)
            + f64x4::splat_t(t, 1.0))
            * a)
            + f64x4::splat_t(t, 1.0);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_index_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let _ = b;
        let r = f64x4::splat_t(t, 1.0 / 6.0)
            .algebraic_mul(a)
            .algebraic_add(f64x4::splat_t(t, 0.5))
            .algebraic_mul(a)
            .algebraic_add(f64x4::splat_t(t, 1.0))
            .algebraic_mul(a)
            .algebraic_add(f64x4::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_token_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let _ = b;
        let r = f64x4::splat_t(t, 1.0 / 6.0)
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f64x4::splat_t(t, 0.5))
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f64x4::splat_t(t, 1.0))
            .algebraic_mul_t(t, a)
            .algebraic_add_t(t, f64x4::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_map_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let _ = b;
        let r = map_add_f64(
            t,
            map_mul_f64(
                t,
                map_add_f64(
                    t,
                    map_mul_f64(
                        t,
                        map_add_f64(
                            t,
                            map_mul_f64(t, f64x4::splat_t(t, 1.0 / 6.0), a),
                            f64x4::splat_t(t, 0.5),
                        ),
                        a,
                    ),
                    f64x4::splat_t(t, 1.0),
                ),
                a,
            ),
            f64x4::splat_t(t, 1.0),
        );
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_auto_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for (o, &a) in out.iter_mut().zip(x) {
        *o = (1.0_f64 / 6.0)
            .algebraic_mul(a)
            .algebraic_add(0.5)
            .algebraic_mul(a)
            .algebraic_add(1.0)
            .algebraic_mul(a)
            .algebraic_add(1.0);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_fused_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let _ = b;
        let r = f64x4::splat_t(t, 1.0 / 6.0)
            .mul_add_fused(a, f64x4::splat_t(t, 0.5))
            .mul_add_fused(a, f64x4::splat_t(t, 1.0))
            .mul_add_fused(a, f64x4::splat_t(t, 1.0));
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum += a * b;
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_index_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum = sum.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_token_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum = sum.algebraic_add_t(t, a.algebraic_mul_t(t, b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_map_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum = map_add_f64(t, sum, map_mul_f64(t, a, b));
    }
    out[0] = sum.algebraic_reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_auto_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut r = 0.0_f64;
    for (&a, &b) in x.iter().zip(y) {
        r = r.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = r;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_fused_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum = a.mul_add_fused(b, sum);
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_index_ordinary_reduce_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let mut sum = f64x4::splat_t(t, 0.0);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        sum = sum.algebraic_add(a.algebraic_mul(b));
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn div_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = a / b;
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn div_index_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = a.algebraic_div(b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn div_token_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = a.algebraic_div_t(t, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn div_map_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = map_div_f64(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn div_auto_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        *o = a.algebraic_div(b);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn rem_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = map_rem_ordinary_f64(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn rem_index_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = a.algebraic_rem(b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn rem_token_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = a.algebraic_rem_t(t, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn rem_map_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, a), b) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        let a = f64x4::load_t(t, a);
        let b = f64x4::load_t(t, b);
        let r = map_rem_f64(t, a, b);
        r.store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn rem_auto_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    let _ = t;
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(y) {
        *o = a.algebraic_rem(b);
    }
}
#[inline(always)]
fn map_add_f64<T: magetypes::simd::backends::F64x4Backend>(
    t: T,
    a: magetypes::simd::generic::f64x4<T>,
    b: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f64x4::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_add(b[i])),
    )
}
#[inline(always)]
fn map_mul_f64<T: magetypes::simd::backends::F64x4Backend>(
    t: T,
    a: magetypes::simd::generic::f64x4<T>,
    b: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f64x4::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_mul(b[i])),
    )
}
#[inline(always)]
fn map_div_f64<T: magetypes::simd::backends::F64x4Backend>(
    t: T,
    a: magetypes::simd::generic::f64x4<T>,
    b: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f64x4::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_div(b[i])),
    )
}
#[inline(always)]
fn map_rem_f64<T: magetypes::simd::backends::F64x4Backend>(
    t: T,
    a: magetypes::simd::generic::f64x4<T>,
    b: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f64x4::from_array_t(
        t,
        core::array::from_fn(|i| a[i].algebraic_rem(b[i])),
    )
}
#[inline(always)]
fn map_rem_ordinary_f64<T: magetypes::simd::backends::F64x4Backend>(
    t: T,
    a: magetypes::simd::generic::f64x4<T>,
    b: magetypes::simd::generic::f64x4<T>,
) -> magetypes::simd::generic::f64x4<T> {
    let a = a.to_array();
    let b = b.to_array();
    magetypes::simd::generic::f64x4::from_array_t(t, core::array::from_fn(|i| a[i] % b[i]))
}
pub fn cases_f32() -> Vec<Case<f32>> {
    let mut v = Vec::with_capacity(128);
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| poly_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| poly_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| poly_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| poly_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| poly_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "index/scalar",
            run: Box::new(move |x, y, o| poly_index_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "index/v3",
            run: Box::new(move |x, y, o| poly_index_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "index/v4x",
            run: Box::new(move |x, y, o| poly_index_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "index/neon",
            run: Box::new(move |x, y, o| poly_index_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| poly_index_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "token/scalar",
            run: Box::new(move |x, y, o| poly_token_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "token/v3",
            run: Box::new(move |x, y, o| poly_token_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "token/v4x",
            run: Box::new(move |x, y, o| poly_token_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "token/neon",
            run: Box::new(move |x, y, o| poly_token_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| poly_token_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "map/scalar",
            run: Box::new(move |x, y, o| poly_map_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "map/v3",
            run: Box::new(move |x, y, o| poly_map_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "map/v4x",
            run: Box::new(move |x, y, o| poly_map_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "map/neon",
            run: Box::new(move |x, y, o| poly_map_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| poly_map_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| poly_auto_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "auto/v3",
            run: Box::new(move |x, y, o| poly_auto_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| poly_auto_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "auto/neon",
            run: Box::new(move |x, y, o| poly_auto_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| poly_auto_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "fused/scalar",
            run: Box::new(move |x, y, o| poly_fused_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "fused/v3",
            run: Box::new(move |x, y, o| poly_fused_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "fused/v4x",
            run: Box::new(move |x, y, o| poly_fused_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "fused/neon",
            run: Box::new(move |x, y, o| poly_fused_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f32",
            name: "fused/wasm128",
            run: Box::new(move |x, y, o| poly_fused_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| dot_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| dot_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| dot_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| dot_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| dot_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index/scalar",
            run: Box::new(move |x, y, o| dot_index_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index/v3",
            run: Box::new(move |x, y, o| dot_index_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index/v4x",
            run: Box::new(move |x, y, o| dot_index_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index/neon",
            run: Box::new(move |x, y, o| dot_index_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| dot_index_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "token/scalar",
            run: Box::new(move |x, y, o| dot_token_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "token/v3",
            run: Box::new(move |x, y, o| dot_token_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "token/v4x",
            run: Box::new(move |x, y, o| dot_token_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "token/neon",
            run: Box::new(move |x, y, o| dot_token_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| dot_token_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "map/scalar",
            run: Box::new(move |x, y, o| dot_map_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "map/v3",
            run: Box::new(move |x, y, o| dot_map_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "map/v4x",
            run: Box::new(move |x, y, o| dot_map_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "map/neon",
            run: Box::new(move |x, y, o| dot_map_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| dot_map_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| dot_auto_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "auto/v3",
            run: Box::new(move |x, y, o| dot_auto_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| dot_auto_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "auto/neon",
            run: Box::new(move |x, y, o| dot_auto_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| dot_auto_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "fused/scalar",
            run: Box::new(move |x, y, o| dot_fused_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "fused/v3",
            run: Box::new(move |x, y, o| dot_fused_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "fused/v4x",
            run: Box::new(move |x, y, o| dot_fused_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "fused/neon",
            run: Box::new(move |x, y, o| dot_fused_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "fused/wasm128",
            run: Box::new(move |x, y, o| dot_fused_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index_ordinary_reduce/scalar",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index_ordinary_reduce/v3",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index_ordinary_reduce/v4x",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index_ordinary_reduce/neon",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f32",
            name: "index_ordinary_reduce/wasm128",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| div_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| div_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| div_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| div_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| div_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "index/scalar",
            run: Box::new(move |x, y, o| div_index_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "index/v3",
            run: Box::new(move |x, y, o| div_index_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "index/v4x",
            run: Box::new(move |x, y, o| div_index_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "index/neon",
            run: Box::new(move |x, y, o| div_index_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| div_index_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "token/scalar",
            run: Box::new(move |x, y, o| div_token_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "token/v3",
            run: Box::new(move |x, y, o| div_token_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "token/v4x",
            run: Box::new(move |x, y, o| div_token_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "token/neon",
            run: Box::new(move |x, y, o| div_token_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| div_token_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "map/scalar",
            run: Box::new(move |x, y, o| div_map_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "map/v3",
            run: Box::new(move |x, y, o| div_map_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "map/v4x",
            run: Box::new(move |x, y, o| div_map_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "map/neon",
            run: Box::new(move |x, y, o| div_map_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| div_map_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| div_auto_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "auto/v3",
            run: Box::new(move |x, y, o| div_auto_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| div_auto_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f32",
            name: "auto/neon",
            run: Box::new(move |x, y, o| div_auto_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f32",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| div_auto_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| rem_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| rem_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| rem_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| rem_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| rem_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "index/scalar",
            run: Box::new(move |x, y, o| rem_index_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "index/v3",
            run: Box::new(move |x, y, o| rem_index_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "index/v4x",
            run: Box::new(move |x, y, o| rem_index_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "index/neon",
            run: Box::new(move |x, y, o| rem_index_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| rem_index_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "token/scalar",
            run: Box::new(move |x, y, o| rem_token_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "token/v3",
            run: Box::new(move |x, y, o| rem_token_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "token/v4x",
            run: Box::new(move |x, y, o| rem_token_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "token/neon",
            run: Box::new(move |x, y, o| rem_token_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| rem_token_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "map/scalar",
            run: Box::new(move |x, y, o| rem_map_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "map/v3",
            run: Box::new(move |x, y, o| rem_map_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "map/v4x",
            run: Box::new(move |x, y, o| rem_map_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "map/neon",
            run: Box::new(move |x, y, o| rem_map_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| rem_map_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| rem_auto_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "auto/v3",
            run: Box::new(move |x, y, o| rem_auto_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| rem_auto_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "auto/neon",
            run: Box::new(move |x, y, o| rem_auto_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f32",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| rem_auto_f32_wasm128_t(t, x, y, o)),
        });
    }
    v
}
pub fn cases_f64() -> Vec<Case<f64>> {
    let mut v = Vec::with_capacity(128);
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| poly_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| poly_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| poly_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| poly_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| poly_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "index/scalar",
            run: Box::new(move |x, y, o| poly_index_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "index/v3",
            run: Box::new(move |x, y, o| poly_index_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "index/v4x",
            run: Box::new(move |x, y, o| poly_index_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "index/neon",
            run: Box::new(move |x, y, o| poly_index_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| poly_index_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "token/scalar",
            run: Box::new(move |x, y, o| poly_token_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "token/v3",
            run: Box::new(move |x, y, o| poly_token_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "token/v4x",
            run: Box::new(move |x, y, o| poly_token_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "token/neon",
            run: Box::new(move |x, y, o| poly_token_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| poly_token_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "map/scalar",
            run: Box::new(move |x, y, o| poly_map_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "map/v3",
            run: Box::new(move |x, y, o| poly_map_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "map/v4x",
            run: Box::new(move |x, y, o| poly_map_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "map/neon",
            run: Box::new(move |x, y, o| poly_map_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| poly_map_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| poly_auto_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "auto/v3",
            run: Box::new(move |x, y, o| poly_auto_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| poly_auto_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "auto/neon",
            run: Box::new(move |x, y, o| poly_auto_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| poly_auto_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "fused/scalar",
            run: Box::new(move |x, y, o| poly_fused_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "fused/v3",
            run: Box::new(move |x, y, o| poly_fused_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "fused/v4x",
            run: Box::new(move |x, y, o| poly_fused_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "fused/neon",
            run: Box::new(move |x, y, o| poly_fused_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "poly/f64",
            name: "fused/wasm128",
            run: Box::new(move |x, y, o| poly_fused_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| dot_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| dot_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| dot_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| dot_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| dot_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index/scalar",
            run: Box::new(move |x, y, o| dot_index_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index/v3",
            run: Box::new(move |x, y, o| dot_index_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index/v4x",
            run: Box::new(move |x, y, o| dot_index_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index/neon",
            run: Box::new(move |x, y, o| dot_index_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| dot_index_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "token/scalar",
            run: Box::new(move |x, y, o| dot_token_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "token/v3",
            run: Box::new(move |x, y, o| dot_token_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "token/v4x",
            run: Box::new(move |x, y, o| dot_token_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "token/neon",
            run: Box::new(move |x, y, o| dot_token_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| dot_token_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "map/scalar",
            run: Box::new(move |x, y, o| dot_map_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "map/v3",
            run: Box::new(move |x, y, o| dot_map_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "map/v4x",
            run: Box::new(move |x, y, o| dot_map_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "map/neon",
            run: Box::new(move |x, y, o| dot_map_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| dot_map_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| dot_auto_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "auto/v3",
            run: Box::new(move |x, y, o| dot_auto_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| dot_auto_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "auto/neon",
            run: Box::new(move |x, y, o| dot_auto_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| dot_auto_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "fused/scalar",
            run: Box::new(move |x, y, o| dot_fused_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "fused/v3",
            run: Box::new(move |x, y, o| dot_fused_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "fused/v4x",
            run: Box::new(move |x, y, o| dot_fused_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "fused/neon",
            run: Box::new(move |x, y, o| dot_fused_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "fused/wasm128",
            run: Box::new(move |x, y, o| dot_fused_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index_ordinary_reduce/scalar",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index_ordinary_reduce/v3",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index_ordinary_reduce/v4x",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index_ordinary_reduce/neon",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "dot/f64",
            name: "index_ordinary_reduce/wasm128",
            run: Box::new(move |x, y, o| dot_index_ordinary_reduce_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| div_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| div_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| div_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| div_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| div_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "index/scalar",
            run: Box::new(move |x, y, o| div_index_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "index/v3",
            run: Box::new(move |x, y, o| div_index_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "index/v4x",
            run: Box::new(move |x, y, o| div_index_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "index/neon",
            run: Box::new(move |x, y, o| div_index_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| div_index_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "token/scalar",
            run: Box::new(move |x, y, o| div_token_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "token/v3",
            run: Box::new(move |x, y, o| div_token_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "token/v4x",
            run: Box::new(move |x, y, o| div_token_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "token/neon",
            run: Box::new(move |x, y, o| div_token_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| div_token_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "map/scalar",
            run: Box::new(move |x, y, o| div_map_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "map/v3",
            run: Box::new(move |x, y, o| div_map_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "map/v4x",
            run: Box::new(move |x, y, o| div_map_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "map/neon",
            run: Box::new(move |x, y, o| div_map_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| div_map_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| div_auto_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "auto/v3",
            run: Box::new(move |x, y, o| div_auto_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| div_auto_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "div/f64",
            name: "auto/neon",
            run: Box::new(move |x, y, o| div_auto_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "div/f64",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| div_auto_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "ordinary/scalar",
            run: Box::new(move |x, y, o| rem_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "ordinary/v3",
            run: Box::new(move |x, y, o| rem_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "ordinary/v4x",
            run: Box::new(move |x, y, o| rem_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "ordinary/neon",
            run: Box::new(move |x, y, o| rem_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "ordinary/wasm128",
            run: Box::new(move |x, y, o| rem_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "index/scalar",
            run: Box::new(move |x, y, o| rem_index_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "index/v3",
            run: Box::new(move |x, y, o| rem_index_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "index/v4x",
            run: Box::new(move |x, y, o| rem_index_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "index/neon",
            run: Box::new(move |x, y, o| rem_index_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "index/wasm128",
            run: Box::new(move |x, y, o| rem_index_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "token/scalar",
            run: Box::new(move |x, y, o| rem_token_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "token/v3",
            run: Box::new(move |x, y, o| rem_token_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "token/v4x",
            run: Box::new(move |x, y, o| rem_token_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "token/neon",
            run: Box::new(move |x, y, o| rem_token_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "token/wasm128",
            run: Box::new(move |x, y, o| rem_token_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "map/scalar",
            run: Box::new(move |x, y, o| rem_map_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "map/v3",
            run: Box::new(move |x, y, o| rem_map_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "map/v4x",
            run: Box::new(move |x, y, o| rem_map_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "map/neon",
            run: Box::new(move |x, y, o| rem_map_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "map/wasm128",
            run: Box::new(move |x, y, o| rem_map_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "auto/scalar",
            run: Box::new(move |x, y, o| rem_auto_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "auto/v3",
            run: Box::new(move |x, y, o| rem_auto_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "auto/v4x",
            run: Box::new(move |x, y, o| rem_auto_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "auto/neon",
            run: Box::new(move |x, y, o| rem_auto_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            group: "rem/f64",
            name: "auto/wasm128",
            run: Box::new(move |x, y, o| rem_auto_f64_wasm128_t(t, x, y, o)),
        });
    }
    v
}

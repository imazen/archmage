//! Algebraic experiments only: no portable-bit or ULP contract.
use crate::Case;
use archmage::prelude::*;
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    let mut sum = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        sum += a * b
    }
    out[0] = sum;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_algebraic_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    let mut sum = 0.0_f32;
    for (&a, &b) in x.iter().zip(y) {
        sum = sum.algebraic_add(a.algebraic_mul(b))
    }
    out[0] = sum;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_ordinary_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    for (o, &a) in out.iter_mut().zip(x) {
        *o = ((0.16666667 * a + 0.5) * a + 1.0) * a + 1.0;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_algebraic_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    for (o, &a) in out.iter_mut().zip(x) {
        *o = 0.16666667_f32
            .algebraic_mul(a)
            .algebraic_add(0.5)
            .algebraic_mul(a)
            .algebraic_add(1.0)
            .algebraic_mul(a)
            .algebraic_add(1.0);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_vector_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let c = |v| f32x8::splat_t(t, v);
        (((c(0.16666667) * a + c(0.5)) * a + c(1.0)) * a + c(1.0)).store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_array_algebraic_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a).to_array();
        let r = a.map(|a| {
            0.16666667_f32
                .algebraic_mul(a)
                .algebraic_add(0.5)
                .algebraic_mul(a)
                .algebraic_add(1.0)
                .algebraic_mul(a)
                .algebraic_add(1.0)
        });
        f32x8::from_array_t(t, r).store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_vector_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    let mut sum = f32x8::splat_t(t, 0.);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0.iter()) {
        sum += f32x8::load_t(t, a) * f32x8::load_t(t, b);
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_array_algebraic_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    let mut sum = [0.0_f32; 8];
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0.iter()) {
        let a = f32x8::load_t(t, a).to_array();
        let b = f32x8::load_t(t, b).to_array();
        sum = core::array::from_fn(|i| sum[i].algebraic_add(a[i].algebraic_mul(b[i])));
    }
    out[0] = sum.into_iter().fold(0.0_f32, |s, a| s.algebraic_add(a));
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn poly_vector_fused_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0.iter())
    {
        let a = f32x8::load_t(t, a);
        let c = |v| f32x8::splat_t(t, v);
        c(0.16666667)
            .mul_add_fused(a, c(0.5))
            .mul_add_fused(a, c(1.0))
            .mul_add_fused(a, c(1.0))
            .store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f32x8))]
pub fn dot_vector_fused_f32(t: Token, x: &[f32], y: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 8, 0);
    let _ = t;
    let _ = y;
    let mut sum = f32x8::splat_t(t, 0.);
    for (a, b) in x.as_chunks::<8>().0.iter().zip(y.as_chunks::<8>().0.iter()) {
        sum = f32x8::load_t(t, a).mul_add_fused(f32x8::load_t(t, b), sum);
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    let mut sum = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        sum += a * b
    }
    out[0] = sum;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_algebraic_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    let mut sum = 0.0_f64;
    for (&a, &b) in x.iter().zip(y) {
        sum = sum.algebraic_add(a.algebraic_mul(b))
    }
    out[0] = sum;
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_ordinary_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    for (o, &a) in out.iter_mut().zip(x) {
        *o = ((0.16666666666666666 * a + 0.5) * a + 1.0) * a + 1.0;
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_algebraic_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    for (o, &a) in out.iter_mut().zip(x) {
        *o = 0.16666666666666666_f64
            .algebraic_mul(a)
            .algebraic_add(0.5)
            .algebraic_mul(a)
            .algebraic_add(1.0)
            .algebraic_mul(a)
            .algebraic_add(1.0);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_vector_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
    {
        let a = f64x4::load_t(t, a);
        let c = |v| f64x4::splat_t(t, v);
        (((c(0.16666666666666666) * a + c(0.5)) * a + c(1.0)) * a + c(1.0)).store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_array_algebraic_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
    {
        let a = f64x4::load_t(t, a).to_array();
        let r = a.map(|a| {
            0.16666666666666666_f64
                .algebraic_mul(a)
                .algebraic_add(0.5)
                .algebraic_mul(a)
                .algebraic_add(1.0)
                .algebraic_mul(a)
                .algebraic_add(1.0)
        });
        f64x4::from_array_t(t, r).store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_vector_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    let mut sum = f64x4::splat_t(t, 0.);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0.iter()) {
        sum += f64x4::load_t(t, a) * f64x4::load_t(t, b);
    }
    out[0] = sum.reduce_add();
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_array_algebraic_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    let mut sum = [0.0_f64; 4];
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0.iter()) {
        let a = f64x4::load_t(t, a).to_array();
        let b = f64x4::load_t(t, b).to_array();
        sum = core::array::from_fn(|i| sum[i].algebraic_add(a[i].algebraic_mul(b[i])));
    }
    out[0] = sum.into_iter().fold(0.0_f64, |s, a| s.algebraic_add(a));
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn poly_vector_fused_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    for (o, a) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0.iter())
    {
        let a = f64x4::load_t(t, a);
        let c = |v| f64x4::splat_t(t, v);
        c(0.16666666666666666)
            .mul_add_fused(a, c(0.5))
            .mul_add_fused(a, c(1.0))
            .mul_add_fused(a, c(1.0))
            .store(o);
    }
}
#[attune(_*_t,+v4x(cfg(avx512)),define(f64x4))]
pub fn dot_vector_fused_f64(t: Token, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % 4, 0);
    let _ = t;
    let _ = y;
    let mut sum = f64x4::splat_t(t, 0.);
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0.iter()) {
        sum = f64x4::load_t(t, a).mul_add_fused(f64x4::load_t(t, b), sum);
    }
    out[0] = sum.reduce_add();
}
pub fn cases_f32() -> Vec<Case<f32>> {
    let mut v = Vec::new();
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/scalar",
            run: Box::new(move |x, y, o| dot_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/v3",
            run: Box::new(move |x, y, o| dot_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/v4x",
            run: Box::new(move |x, y, o| dot_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/neon",
            run: Box::new(move |x, y, o| dot_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/wasm128",
            run: Box::new(move |x, y, o| dot_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/scalar",
            run: Box::new(move |x, y, o| dot_algebraic_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/v3",
            run: Box::new(move |x, y, o| dot_algebraic_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/v4x",
            run: Box::new(move |x, y, o| dot_algebraic_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/neon",
            run: Box::new(move |x, y, o| dot_algebraic_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/wasm128",
            run: Box::new(move |x, y, o| dot_algebraic_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/scalar",
            run: Box::new(move |x, y, o| poly_ordinary_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/v3",
            run: Box::new(move |x, y, o| poly_ordinary_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/v4x",
            run: Box::new(move |x, y, o| poly_ordinary_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/neon",
            run: Box::new(move |x, y, o| poly_ordinary_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/wasm128",
            run: Box::new(move |x, y, o| poly_ordinary_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/scalar",
            run: Box::new(move |x, y, o| poly_algebraic_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/v3",
            run: Box::new(move |x, y, o| poly_algebraic_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/v4x",
            run: Box::new(move |x, y, o| poly_algebraic_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/neon",
            run: Box::new(move |x, y, o| poly_algebraic_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/wasm128",
            run: Box::new(move |x, y, o| poly_algebraic_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/scalar",
            run: Box::new(move |x, y, o| poly_vector_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/v3",
            run: Box::new(move |x, y, o| poly_vector_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/v4x",
            run: Box::new(move |x, y, o| poly_vector_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/neon",
            run: Box::new(move |x, y, o| poly_vector_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/wasm128",
            run: Box::new(move |x, y, o| poly_vector_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/scalar",
            run: Box::new(move |x, y, o| poly_array_algebraic_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/v3",
            run: Box::new(move |x, y, o| poly_array_algebraic_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/v4x",
            run: Box::new(move |x, y, o| poly_array_algebraic_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/neon",
            run: Box::new(move |x, y, o| poly_array_algebraic_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/wasm128",
            run: Box::new(move |x, y, o| poly_array_algebraic_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/scalar",
            run: Box::new(move |x, y, o| dot_vector_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/v3",
            run: Box::new(move |x, y, o| dot_vector_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/v4x",
            run: Box::new(move |x, y, o| dot_vector_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/neon",
            run: Box::new(move |x, y, o| dot_vector_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/wasm128",
            run: Box::new(move |x, y, o| dot_vector_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/scalar",
            run: Box::new(move |x, y, o| dot_array_algebraic_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/v3",
            run: Box::new(move |x, y, o| dot_array_algebraic_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/v4x",
            run: Box::new(move |x, y, o| dot_array_algebraic_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/neon",
            run: Box::new(move |x, y, o| dot_array_algebraic_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/wasm128",
            run: Box::new(move |x, y, o| dot_array_algebraic_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/scalar",
            run: Box::new(move |x, y, o| poly_vector_fused_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/v3",
            run: Box::new(move |x, y, o| poly_vector_fused_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/v4x",
            run: Box::new(move |x, y, o| poly_vector_fused_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/neon",
            run: Box::new(move |x, y, o| poly_vector_fused_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/wasm128",
            run: Box::new(move |x, y, o| poly_vector_fused_f32_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/scalar",
            run: Box::new(move |x, y, o| dot_vector_fused_f32_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/v3",
            run: Box::new(move |x, y, o| dot_vector_fused_f32_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/v4x",
            run: Box::new(move |x, y, o| dot_vector_fused_f32_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/neon",
            run: Box::new(move |x, y, o| dot_vector_fused_f32_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/wasm128",
            run: Box::new(move |x, y, o| dot_vector_fused_f32_wasm128_t(t, x, y, o)),
        });
    }
    v
}
pub fn cases_f64() -> Vec<Case<f64>> {
    let mut v = Vec::new();
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/scalar",
            run: Box::new(move |x, y, o| dot_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/v3",
            run: Box::new(move |x, y, o| dot_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/v4x",
            run: Box::new(move |x, y, o| dot_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/neon",
            run: Box::new(move |x, y, o| dot_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_ordinary",
            label: "dot_ordinary/wasm128",
            run: Box::new(move |x, y, o| dot_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/scalar",
            run: Box::new(move |x, y, o| dot_algebraic_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/v3",
            run: Box::new(move |x, y, o| dot_algebraic_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/v4x",
            run: Box::new(move |x, y, o| dot_algebraic_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/neon",
            run: Box::new(move |x, y, o| dot_algebraic_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_algebraic",
            label: "dot_algebraic/wasm128",
            run: Box::new(move |x, y, o| dot_algebraic_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/scalar",
            run: Box::new(move |x, y, o| poly_ordinary_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/v3",
            run: Box::new(move |x, y, o| poly_ordinary_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/v4x",
            run: Box::new(move |x, y, o| poly_ordinary_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/neon",
            run: Box::new(move |x, y, o| poly_ordinary_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_ordinary",
            label: "poly_ordinary/wasm128",
            run: Box::new(move |x, y, o| poly_ordinary_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/scalar",
            run: Box::new(move |x, y, o| poly_algebraic_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/v3",
            run: Box::new(move |x, y, o| poly_algebraic_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/v4x",
            run: Box::new(move |x, y, o| poly_algebraic_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/neon",
            run: Box::new(move |x, y, o| poly_algebraic_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_algebraic",
            label: "poly_algebraic/wasm128",
            run: Box::new(move |x, y, o| poly_algebraic_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/scalar",
            run: Box::new(move |x, y, o| poly_vector_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/v3",
            run: Box::new(move |x, y, o| poly_vector_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/v4x",
            run: Box::new(move |x, y, o| poly_vector_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/neon",
            run: Box::new(move |x, y, o| poly_vector_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_vector",
            label: "poly_vector/wasm128",
            run: Box::new(move |x, y, o| poly_vector_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/scalar",
            run: Box::new(move |x, y, o| poly_array_algebraic_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/v3",
            run: Box::new(move |x, y, o| poly_array_algebraic_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/v4x",
            run: Box::new(move |x, y, o| poly_array_algebraic_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/neon",
            run: Box::new(move |x, y, o| poly_array_algebraic_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_array_algebraic",
            label: "poly_array_algebraic/wasm128",
            run: Box::new(move |x, y, o| poly_array_algebraic_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/scalar",
            run: Box::new(move |x, y, o| dot_vector_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/v3",
            run: Box::new(move |x, y, o| dot_vector_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/v4x",
            run: Box::new(move |x, y, o| dot_vector_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/neon",
            run: Box::new(move |x, y, o| dot_vector_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_vector",
            label: "dot_vector/wasm128",
            run: Box::new(move |x, y, o| dot_vector_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/scalar",
            run: Box::new(move |x, y, o| dot_array_algebraic_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/v3",
            run: Box::new(move |x, y, o| dot_array_algebraic_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/v4x",
            run: Box::new(move |x, y, o| dot_array_algebraic_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/neon",
            run: Box::new(move |x, y, o| dot_array_algebraic_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_array_algebraic",
            label: "dot_array_algebraic/wasm128",
            run: Box::new(move |x, y, o| dot_array_algebraic_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/scalar",
            run: Box::new(move |x, y, o| poly_vector_fused_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/v3",
            run: Box::new(move |x, y, o| poly_vector_fused_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/v4x",
            run: Box::new(move |x, y, o| poly_vector_fused_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/neon",
            run: Box::new(move |x, y, o| poly_vector_fused_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "poly_vector_fused",
            label: "poly_vector_fused/wasm128",
            run: Box::new(move |x, y, o| poly_vector_fused_f64_wasm128_t(t, x, y, o)),
        });
    }
    if let Some(t) = ScalarToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/scalar",
            run: Box::new(move |x, y, o| dot_vector_fused_f64_scalar_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/v3",
            run: Box::new(move |x, y, o| dot_vector_fused_f64_v3_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4xToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/v4x",
            run: Box::new(move |x, y, o| dot_vector_fused_f64_v4x_t(t, x, y, o)),
        });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/neon",
            run: Box::new(move |x, y, o| dot_vector_fused_f64_neon_t(t, x, y, o)),
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = Wasm128Token::summon() {
        v.push(Case {
            op: "dot_vector_fused",
            label: "dot_vector_fused/wasm128",
            run: Box::new(move |x, y, o| dot_vector_fused_f64_wasm128_t(t, x, y, o)),
        });
    }
    v
}

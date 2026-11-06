//! Adapters to Zenmath's shared intrinsic-only CORE-MATH implementation.
use magetypes::simd::backends::F64x4Backend;
#[inline(always)]
pub(crate) fn exp<T: F64x4Backend + archmage::IntoConcreteToken>(t: T, x: [f32; 4]) -> [f32; 4] {
    archmage::attuned!(
        zenmath::simd::array::exp_f32x4(x),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    )
}
#[inline(always)]
pub(crate) fn log<T: F64x4Backend + archmage::IntoConcreteToken, const TWO: bool>(
    t: T,
    x: [f32; 4],
) -> [f32; 4] {
    if TWO {
        archmage::attuned!(
            zenmath::simd::array::log2_f32x4(x),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        )
    } else {
        archmage::attuned!(
            zenmath::simd::array::ln_f32x4(x),
            [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
            using(t)
        )
    }
}
#[inline(always)]
pub(crate) fn pow<T: F64x4Backend + archmage::IntoConcreteToken>(
    t: T,
    x: [f32; 4],
    y: [f32; 4],
) -> [f32; 4] {
    archmage::attuned!(
        zenmath::simd::array::pow_f32x4(x, y),
        [_v3, _neon, _wasm128, _scalar, _v4x(cfg(avx512))],
        using(t)
    )
}

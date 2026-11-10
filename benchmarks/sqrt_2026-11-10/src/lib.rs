//! Compile-only probes, separate from timed kernels and shipped library exports.
#![no_std]
#![forbid(unsafe_code)]

#[inline(never)]
pub fn sqrt_probe_scalar_f32(x: f32) -> f32 {
    magetypes::simd::scalar::f32x1(x).sqrt().0
}

#[inline(never)]
pub fn sqrt_probe_scalar_f64(x: f64) -> f64 {
    magetypes::simd::scalar::f64x1(x).sqrt().0
}

#[inline(never)]
pub fn sqrt_probe_scalar_f32x4(x: [f32; 4]) -> [f32; 4] {
    magetypes::simd::generic::f32x4::from_array_t(archmage::ScalarToken, x)
        .sqrt()
        .to_array()
}

#[inline(never)]
pub fn sqrt_probe_scalar_f64x2(x: [f64; 2]) -> [f64; 2] {
    magetypes::simd::generic::f64x2::from_array_t(archmage::ScalarToken, x)
        .sqrt()
        .to_array()
}

// Verify that ordinary relaxed builds retain their existing opcode selection,
// while testing flags are absent. Array arguments keep this tool's API simple.
#[cfg(target_arch = "wasm32")]
#[inline(never)]
pub fn fma_probe_relaxed_f32(
    t: archmage::Wasm128Token,
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
) -> [f32; 4] {
    use magetypes::simd::generic::f32x4 as V;
    V::from_array_t(t, a)
        .mul_add_relaxed(V::from_array_t(t, b), V::from_array_t(t, c))
        .to_array()
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
pub fn fma_probe_relaxed_f64(
    t: archmage::Wasm128Token,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
) -> [f64; 2] {
    use magetypes::simd::generic::f64x2 as V;
    V::from_array_t(t, a)
        .mul_add_relaxed(V::from_array_t(t, b), V::from_array_t(t, c))
        .to_array()
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
pub fn fma_probe_fused_f32(
    t: archmage::Wasm128Token,
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
) -> [f32; 4] {
    use magetypes::simd::generic::f32x4 as V;
    V::from_array_t(t, a)
        .mul_add_fused(V::from_array_t(t, b), V::from_array_t(t, c))
        .to_array()
}

#[cfg(target_arch = "wasm32")]
#[inline(never)]
pub fn fma_probe_fused_f64(
    t: archmage::Wasm128Token,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
) -> [f64; 2] {
    use magetypes::simd::generic::f64x2 as V;
    V::from_array_t(t, a)
        .mul_add_fused(V::from_array_t(t, b), V::from_array_t(t, c))
        .to_array()
}
